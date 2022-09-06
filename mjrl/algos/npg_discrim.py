import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
# utility functions
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve
from mjrl.algos.npg_cg import NPG
from torch.utils.tensorboard import SummaryWriter

class NPGDiscriminator(NPG):
    def __init__(self, env, policy, baseline, discriminator,
                 frame_num=1,
                 state_only=False,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 input_normalization=None,
                 log_dir=None,
                 **kwargs
                 ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.discriminator = discriminator
        self.frame_num = frame_num
        self.state_only = state_only
        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.demo_paths = demo_paths
        if save_logs: self.logger = DataLog()
        # input normalization (running average)
        self.input_normalization = input_normalization
        if self.input_normalization is not None:
            if self.input_normalization > 1 or self.input_normalization <= 0:
                self.input_normalization = None
        self.writer = SummaryWriter(f"runs/{log_dir}")
        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

    def sample_processing(self, obs, act):
        if self.env.env_id == 'door-v0':
            process_obs = obs[:, 3:27]
            process_act = act[:, 4:]   
        elif self.env.env_id == 'relocate-v0':
            process_obs = obs[:, 6:30]
            process_act = act[:, 6:]
        elif self.env.env_id == 'pen-v0':
            process_obs = obs[:, :24]
            process_act = act         
        elif self.env.env_id == 'hammer-v0':
            process_obs = obs[:, 2:26]
            process_act = act[:, 2:]
        else: raise NotImplementedError
        return self.framestack(process_obs, process_act)


    def framestack(self, obs, act):
        stack_obs = np.array([obs[i:i+self.frame_num] for i in range(obs.shape[0]-self.frame_num+1)]) # shape: N to N-num+1
        if self.state_only:
            processed_sample = stack_obs.reshape(stack_obs.shape[0], -1)
        else:
            stack_act = np.array([act[i:i+self.frame_num] for i in range(act.shape[0]-self.frame_num+1)]) # shape: N to N-num+1
            stack_obs_act = np.concatenate((stack_obs, stack_act), axis=-1)
            processed_sample = stack_obs_act.reshape(stack_obs_act.shape[0], -1)  # [[sasa...][sasa...]...]
        
        return processed_sample

    def train_from_paths(self, paths):

        # Load demonstrations
        if self.demo_paths is not None:
            demo_obs = np.concatenate([path["observations"] for path in self.demo_paths])
            demo_act = np.concatenate([path["actions"] for path in self.demo_paths])
            
        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        Tensor = torch.FloatTensor

        # Train discriminator, ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
        # demo_sample =  Variable(Tensor(np.concatenate([demo_obs, demo_act], axis=-1)))
        # generated_sample =  Variable(Tensor(np.concatenate([observations, actions], axis=-1)))

        demo_sample = Variable(Tensor(self.sample_processing(demo_obs, demo_act)))
        generated_sample = Variable(Tensor(self.sample_processing(observations, actions)))

        real = Variable(torch.FloatTensor(demo_sample.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(generated_sample.size(0), 1).fill_(0.0), requires_grad=False)
        self.optimizer.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(demo_sample), real)
        fake_loss = self.adversarial_loss(self.discriminator(generated_sample), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer.step()

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # normalize inputs if necessary
        if self.input_normalization:
            data_in_shift, data_in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            pi_in_shift, pi_in_scale = self.policy.model.in_shift.data.numpy(), self.policy.model.in_scale.data.numpy()
            pi_out_shift, pi_out_scale = self.policy.model.out_shift.data.numpy(), self.policy.model.out_scale.data.numpy()
            pi_in_shift = self.input_normalization * pi_in_shift + (1-self.input_normalization) * data_in_shift
            pi_in_scale = self.input_normalization * pi_in_scale + (1-self.input_normalization) * data_in_scale
            self.policy.model.set_transformations(pi_in_shift, pi_in_scale, pi_out_shift, pi_out_scale)

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('discriminator_loss', d_loss.detach().numpy())
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)

            self.writer.add_scalar(f"metric/discriminator_loss", d_loss, self.itr)
            self.writer.add_scalar(f"metric/alpha", alpha, self.itr)
            self.writer.add_scalar(f"metric/delta", n_step_size, self.itr)
            self.writer.add_scalar(f"metric/time_vpg", t_gLL, self.itr)
            self.writer.add_scalar(f"metric/time_npg", t_FIM, self.itr)
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
