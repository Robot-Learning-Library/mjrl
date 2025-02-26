"""
Basic reinforce algorithm using on-policy rollouts
Also has function to perform linesearch on KL (improves stability)
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from torch.utils.tensorboard import SummaryWriter


class BatchREINFORCE:
    def __init__(self, env, policy, baseline,
                 learn_rate=0.01,
                 seed=123,
                 desired_kl=None,
                 save_logs=False,
                 log_dir=None,
                 **kwargs
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        self.desired_kl = desired_kl
        if save_logs: self.logger = DataLog()
        self.writer = SummaryWriter(f"runs/{log_dir}")

    def CPI_surrogate(self, observations, actions, advantages):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        # add entropy boosting
        # policy_entropy = torch.sum(torch.square(torch.exp(self.policy.log_std))) # for multivariate Gaussian distribution, entropy is ~ \sum_i sigma_i^2
        # objective = cpi_surr + 0.001 * policy_entropy
        # vpg_grad = torch.autograd.grad(objective, self.policy.trainable_params)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    # ----------------------------------------------------------
    def train_step(self, N,
                   env=None,
                   sample_mode='trajectories',
                   horizon=1e6,
                   gamma=0.995,
                   gae_lambda=0.97,
                   num_cpu='max',
                   env_kwargs=None,
                   parser_args=None,
                   itr=0,
                   ):
        self.itr = itr
        # Clean up input arguments
        env = self.env.env_id if env is None else env
        if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = trajectory_sampler.sample_paths(**input_dict, parser_args=parser_args)
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = trajectory_sampler.sample_data_batch(**input_dict, parser_args=parser_args)

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed
        
        if self.itr >= int(parser_args.warm_up):
            if self.discriminator_reward:
                self.add_reg_reward(paths)

        if len(paths) > 0:
            # compute returns
            process_samples.compute_returns(paths, gamma)
            # compute advantages
            process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
            # train from paths
            eval_statistics = self.train_from_paths(paths)
            eval_statistics.append(N)
            # log number of samples
            if self.save_logs:
                num_samples = np.sum([p["rewards"].shape[0] for p in paths])
                self.logger.log_kv('num_samples', num_samples)
            # fit baseline
            if self.save_logs:
                ts = timer.time()
                error_before, error_after = self.baseline.fit(paths, return_errors=True)
                self.logger.log_kv('time_VF', timer.time()-ts)
                self.logger.log_kv('VF_error_before', error_before)
                self.logger.log_kv('VF_error_after', error_after)

                self.writer.add_scalar(f"metric/time_VF", timer.time()-ts, self.itr)
                self.writer.add_scalar(f"metric/VF_error_before", error_before, self.itr)
                self.writer.add_scalar(f"metric/VF_error_after", error_after, self.itr)

            else:
                self.baseline.fit(paths)

            return eval_statistics
        else:
            return None

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # Policy update with linesearch
        # ------------------------------
        if self.desired_kl is not None:
            max_ctr = 100
            alpha = self.alpha
            curr_params = self.policy.get_param_values()
            for ctr in range(max_ctr):
                new_params = curr_params + alpha * vpg_grad
                self.policy.set_param_values(new_params, set_new=True, set_old=False)
                kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
                if kl_dist <= self.desired_kl:
                    break
                else:
                    print("backtracking")
                    alpha = alpha / 2.0
        else:
            curr_params = self.policy.get_param_values()
            new_params = curr_params + self.alpha * vpg_grad

        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)

            self.writer.add_scalar(f"metric/alpha", self.alpha, self.itr)
            self.writer.add_scalar(f"metric/time_vpg", t_gLL, self.itr)
            self.writer.add_scalar(f"metric/kl_dist", kl_dist, self.itr)
            self.writer.add_scalar(f"metric/surr_improvement", surr_after - surr_before, self.itr)
            self.writer.add_scalar(f"metric/running_score", self.running_score, self.itr)

            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                    self.writer.add_scalar(f"metric/success_rate", success_rate, self.itr)

                except:
                    pass

        return base_stats


    def process_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        running_score = mean_return if self.running_score is None else \
                        0.9 * self.running_score + 0.1 * mean_return

        return observations, actions, advantages, base_stats, running_score


    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)

        self.writer.add_scalar(f"metric/stoc_pol_mean", mean_return, self.itr)
        self.writer.add_scalar(f"metric/stoc_pol_std", std_return, self.itr)
        self.writer.add_scalar(f"metric/stoc_pol_max", max_return, self.itr)
        self.writer.add_scalar(f"metric/stoc_pol_min", min_return, self.itr)

        try:
            success_rate = self.env.env.env.evaluate_success(paths)
            self.logger.log_kv('rollout_success', success_rate)
            self.writer.add_scalar(f"metric/rollout_success", success_rate, self.itr)
        except:
            pass
