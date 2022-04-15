import numpy as np
import os
import json
import time
import datetime
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))
from GEM_envs import Env_Generator
from Buffer import Replay_Buffer
from Rollouts import Rollout
from Policies import TD3_Policy
from Reward_Functions import Scaled_Reward

UPDATE_DIVIDER_TRAIN = 5

class MRL:
    def __init__(self,
                 train_path,
                 test_path,
                 buffer_size,
                 policy_params,
                 train_batch_size,
                 save_path,
                 use_context=True,
                 name=None,
                ):

        reward_function = Scaled_Reward(gamma=policy_params['gamma'])
        self.use_context = use_context

        self.env_generator = Env_Generator(train_path, test_path)
        sizes = self.env_generator.get_sizes()
        self.num_train = sizes['Train']
        self.num_test = sizes['Test']

        self.train_buffer = Replay_Buffer(buffer_size, sizes['Train'])
        self.eval_buffer = Replay_Buffer(buffer_size, sizes['Test'])
        self.train_rollout = Rollout(self.train_buffer, reward_function)
        self.eval_rollout = Rollout(self.eval_buffer, reward_function)
        self.td3_policy = self.init_policy(policy_params)
        self.train_batch_size = train_batch_size

        curr_datetime = datetime.datetime.now()
        self.load_path = save_path
        if name is None:
            self.save_path = save_path / str(curr_datetime)
        else:
            self.save_path = save_path / str(name)

        self.train_reward_hist = []
        self.policy_params = policy_params
        self.context_input_size = policy_params['context_input_size']

    def init_checkpoints(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.policy_params["train_batch_size"] = self.train_batch_size
        self.save_hyperparams()


    def init_policy(self, policy_params):
        td3_policy = TD3_Policy(
            state_dim = policy_params['state_dim'],
            action_dim=policy_params['action_dim'],
            actor_hidden=policy_params['actor_hidden'],
            critic_hidden=policy_params['critic_hidden'],
            context_hidden=policy_params['context_hidden'],
            actor_lr=policy_params['actor_lr'],
            critic_lr=policy_params['critic_lr'],
            exploration_noise=policy_params['exploration_noise'],
            exploration_noise_clip=policy_params['exploration_noise_clip'],
            gamma=policy_params['gamma'],
            tau=policy_params['tau'],
            policy_noise=policy_params['policy_noise'],
            policy_noise_clip=policy_params['policy_noise_clip'],
            policy_freq=policy_params['policy_freq'],
            use_context = self.use_context,
            context_size=policy_params['context_size'],
            context_input_size=policy_params['context_input_size'],
        )
        return td3_policy

    def train(self, num_steps_task, num_steps_update, task_set, motor_id):
        rollout = self.train_rollout
        batch_size = self.train_batch_size
        task_env = self.env_generator.get_env(motor_id, task_set)
        rollout.rollout(task_env,self.td3_policy,num_steps_task,motor_id,use_context=self.use_context,
                        context_input_size=self.context_input_size)
        print("Mean reward of motor rollout: ", round(np.mean(rollout.last_rollout_rewards), 4))
        self.train_reward_hist.append(rollout.last_rollout_rewards)
        self.td3_policy.update(rollout.replay_buffer,batch_size, num_steps_update)

    def meta_train(self, num_steps_task, num_steps_train, checkpoint_steps=int(1e5)):
        self.init_checkpoints()
        if self.use_context:
            dessca_load = self.load_path / "DESSCA_Samples" / "context_samples_training.npy"
            dessca_samples = np.load(dessca_load)
            dessca_states = dessca_samples[:,:,:4]
            dessca_states= dessca_states[:,:,[2,0,1,3]] # change order such that omega is first
            dessca_actions = dessca_samples[:,:,4:]
            for motor_id_train in range(self.num_train):
                env = self.env_generator.get_env(motor_id_train, "Train")
                self.train_rollout.initial_rollout(env, motor_id_train, dessca_states[motor_id_train],dessca_actions[motor_id_train])
        print("Initialization done")
        self.policy_params['num_steps_task'] = num_steps_task
        self.policy_params['num_steps_train'] = num_steps_train
        self.policy_params['checkpoint_steps'] = checkpoint_steps

        self.start_time = time.process_time()
        curr_step = 0
        nxt_ckpt = checkpoint_steps
        while curr_step < num_steps_train:
            for motor_id_train in range(self.num_train):
                print("Updating step", curr_step)
                self.train(num_steps_task, num_steps_task//UPDATE_DIVIDER_TRAIN, "Train", motor_id_train)
                curr_step += num_steps_task
                if curr_step >= nxt_ckpt:
                    time_proc = time.process_time() - self.start_time
                    nxt_ckpt = curr_step + checkpoint_steps
                    self.create_checkpoint(curr_step, time_proc)
                if curr_step >= num_steps_train:
                    break
        time_proc = time.process_time() - self.start_time
        self.create_checkpoint(curr_step, time_proc)

    def create_checkpoint(self, num_steps, time_proc):
        path = self.save_path / str(num_steps)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save(path)
        info_dict = dict()
        info_dict['Passed Time'] = time_proc
        with open(path / 'training_rewards', 'wb') as storage_file:
            pickle.dump(self.train_reward_hist, storage_file)

        with open(path / 'info_dump.json', 'w') as info_dump_file:
            json.dump(info_dict,info_dump_file)

    def save_hyperparams(self):
        with open(self.save_path / 'hyperparams.json', 'w') as hf:
            json.dump(self.policy_params,hf)

    def save(self, path):
        self.td3_policy.save_networks(path)

    def load(self, target=False):
        self.td3_policy.load_networks(self.load_path, target)
