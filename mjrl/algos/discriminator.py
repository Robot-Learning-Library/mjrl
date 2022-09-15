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
from mjrl.utils.fc_network import FCNetwork

from torch.utils.tensorboard import SummaryWriter

class Discriminator():
    def __init__(self,
                 hidden_size=(128, 128),
                 frame_num=3,
                 state_only=True,
                 itr = 100,
                 save_logs=False,
                 input_normalization=None,
                 log_dir=None,
                 **kwargs
                 ):
        """
        """
        self.frame_num = frame_num
        self.state_only = state_only
        self.save_logs = save_logs
        hand_dim = 24
        if state_only:
            self.model = FCNetwork(frame_num*hand_dim, 1, hidden_sizes=hidden_size, output_nonlinearity='sigmoid')
        else:
            self.model = FCNetwork(frame_num*2*hand_dim, 1, hidden_sizes=hidden_size, output_nonlinearity='sigmoid')
        self.itr = itr

        self.input_normalization = input_normalization
        if self.input_normalization is not None:
            if self.input_normalization > 1 or self.input_normalization <= 0:
                self.input_normalization = None
        self.writer = SummaryWriter(f"runs/{log_dir}")
        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.ture_samples = None
        self.fake_samples = None

    def process_data(self, env_id, true_paths, fake_paths,):
        # Load samples
        if true_paths is not None:
            true_obs = np.concatenate([path["observations"] for path in true_paths])
            true_act = np.concatenate([path["actions"] for path in true_paths])
        if fake_paths is not None:
            fake_obs = np.concatenate([path["observations"] for path in fake_paths])
            fake_act = np.concatenate([path["actions"] for path in fake_paths])

        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        Tensor = torch.FloatTensor
        ture_sample = Variable(Tensor(self.sample_processing(env_id, true_obs, true_act)))
        fake_sample = Variable(Tensor(self.sample_processing(env_id, fake_obs, fake_act)))

        if self.ture_samples is None:
            self.ture_samples = ture_sample
        else:
            self.true_samples = torch.cat([self.true_samples, ture_sample])
        if self.fake_samples is None:
            self.fake_samples = fake_sample
        else:
            self.fake_samples = torch.cat([self.fake_samples, fake_sample])


    def sample_processing(self, env_id, obs, act):
        if env_id == 'door-v0':
            process_obs = obs[:, 3:27]
            process_act = act[:, 4:]   
        elif env_id == 'relocate-v0':
            process_obs = obs[:, 6:30]
            process_act = act[:, 6:]
        elif env_id == 'pen-v0':
            process_obs = obs[:, :24]
            process_act = act         
        elif env_id == 'hammer-v0':
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

    def train(self, ):
        real = Variable(torch.FloatTensor(self.ture_samples.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(self.fake_samples.size(0), 1).fill_(0.0), requires_grad=False)

        for i in range(self.itr):
            # add a sampling scheme
            self.optimizer.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.adversarial_loss(self.model(self.ture_samples), real)
            fake_loss = self.adversarial_loss(self.model(self.fake_samples), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.optimizer.step()

            # Log information
            if self.save_logs:
                self.logger.log_kv('discriminator_loss', d_loss.detach().numpy())
                self.writer.add_scalar(f"metric/discriminator_loss", d_loss, i)

            print(f"Step: {i}  |  Discriminator loss: {d_loss}")
