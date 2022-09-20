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
            self.feature = FCNetwork(frame_num*hand_dim, hidden_size[-1], hidden_sizes=hidden_size, output_nonlinearity='sigmoid')
        else:
            self.feature = FCNetwork(frame_num*2*hand_dim, hidden_size[-1], hidden_sizes=hidden_size, output_nonlinearity='sigmoid')
        self.discriminator = FCNetwork(hidden_size[-1], 1, hidden_sizes=hidden_size, output_nonlinearity='sigmoid')
        self.classifier = FCNetwork(hidden_size[-1], 4, hidden_sizes=hidden_size, output_nonlinearity='softmax')

        self.itr = itr

        self.input_normalization = input_normalization
        if self.input_normalization is not None:
            if self.input_normalization > 1 or self.input_normalization <= 0:
                self.input_normalization = None
        self.writer = SummaryWriter(f"runs/{log_dir}")
        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
        self.classify_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(list(self.feature.parameters()) 
        + list(self.discriminator.parameters())
        + list(self.classifier.parameters())
        , lr=1e-4)

        self.true_samples = None
        self.fake_samples = None
        self.task_labels = None

    def model(self, x):
        feature = self.feature(x)
        x = self.discriminator(feature)
        return x

    def classify(self, x):
        feature = self.feature(x)
        x = self.classifier(feature)
        return x

    def process_data(self, env_id, true_paths, fake_paths,):
        use_latest_paths = 100

        # Load samples
        if true_paths is not None:
            true_obs = np.concatenate([path["observations"] for path in true_paths])
            true_act = np.concatenate([path["actions"] for path in true_paths])
        if fake_paths is not None:
            fake_obs = np.concatenate([path["observations"] for path in fake_paths[-use_latest_paths:]])
            fake_act = np.concatenate([path["actions"] for path in fake_paths[-use_latest_paths:]])

        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        Tensor = torch.FloatTensor
        ture_sample, _ = self.sample_processing(env_id, true_obs, true_act)
        fake_sample, task_label = self.sample_processing(env_id, fake_obs, fake_act)
        ture_sample = Variable(Tensor(ture_sample))
        fake_sample = Variable(Tensor(fake_sample))
        task_labels = Variable(torch.FloatTensor(fake_sample.size(0) * [task_label]), requires_grad=False)

        if self.true_samples is None:
            self.true_samples = ture_sample
        else:
            self.true_samples = torch.cat([self.true_samples, ture_sample])
        if self.fake_samples is None:
            self.fake_samples = fake_sample
        else:
            self.fake_samples = torch.cat([self.fake_samples, fake_sample])
        if self.task_labels is None:
            self.task_labels = task_labels
        else:
            self.task_labels = torch.cat([self.task_labels, task_labels])

    def sample_processing(self, env_id, obs, act):
        if env_id == 'relocate-v0':
            process_obs = obs[:, 6:30]
            process_act = act[:, 6:]
            task_label = [1,0,0,0]
        elif env_id == 'pen-v0':
            process_obs = obs[:, :24]
            process_act = act
            task_label = [0,1,0,0]
        elif env_id == 'hammer-v0':
            process_obs = obs[:, 2:26]
            process_act = act[:, 2:]
            task_label = [0,0,1,0]
        elif env_id == 'door-v0':
            process_obs = obs[:, 3:27]
            process_act = act[:, 4:] 
            task_label = [0,0,0,1]  
        else: raise NotImplementedError
        return self.framestack(process_obs, process_act), task_label


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
        real = Variable(torch.FloatTensor(self.true_samples.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(self.fake_samples.size(0), 1).fill_(0.0), requires_grad=False)

        for i in range(self.itr):
            # add a sampling scheme
            self.optimizer.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = self.adversarial_loss(self.model(self.true_samples), real)
            fake_loss = self.adversarial_loss(self.model(self.fake_samples), fake)
            d_loss = (real_loss + fake_loss) / 2

            # domain classifier
            c_loss = self.classify_loss(self.classify(self.fake_samples), self.task_labels)
            # inverse gradient 
            
            loss =  d_loss + c_loss
            loss.backward()
            self.optimizer.step()

            # Log information
            if self.save_logs:
                self.logger.log_kv('discriminator_loss', d_loss.detach().numpy())
                self.logger.log_kv('classifier_loss', c_loss.detach().numpy())
                self.writer.add_scalar(f"metric/discriminator_loss", d_loss, i)
                self.writer.add_scalar(f"metric/classifier_loss", c_loss, i)

            print(f"Step: {i}  |  Discriminator loss: {d_loss} |  Classifier loss: {c_loss}")
