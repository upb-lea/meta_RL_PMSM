import numpy as np
import torch

class Rollout:
    def __init__(self, replay_buffer, reward_function):
        self.replay_buffer = replay_buffer
        self.last_rollout_rewards = []
        self.reward_function = reward_function

    def rollout(self, env, policy, num_steps, motor_id, det=False, single_episode=False,
                use_context=True, context_input_size=1000):
        print(f"Rollout on motor_id {motor_id}")
        self.last_rollout_rewards = []
        curr_step = 0
        Dones = 0
        done = False
        while (curr_step < num_steps) & ( ~single_episode | (single_episode & ~done)):
            Dones = Dones +1
            done = False
            obs = env.reset()
            while not done and curr_step < num_steps:
                if use_context:
                    transitions = self.replay_buffer.sample(context_input_size, motor_id)
                    transitions = torch.FloatTensor(transitions)
                    transitions = torch.unsqueeze(transitions, 0)
                else:
                    transitions = None
                action = policy.act(torch.FloatTensor([obs]), transitions, det)
                next_obs, reward, done, _ = env.step(action)
                reward = self.reward_function.calc_reward(obs, next_obs)
                self.last_rollout_rewards.append(reward)
                self.replay_buffer.add((obs.tolist().copy(), next_obs.tolist().copy(), action.tolist().copy(), [reward]),motor_id)
                obs = next_obs.copy()
                curr_step = curr_step + 1
        print("Number of episodes in rollout: ", Dones)

    def initial_rollout(self, env, motor_id, states, actions):
        step=1
        for state,action in zip(states, actions):
            print(step, end="\r")
            #States are omega, i_d, i_q, epsilon
            env.reset()
            omega_lim = env.physical_system.limits[0]
            current_lim = env.physical_system.limits[2]
            scale = np.array([omega_lim, current_lim, current_lim, np.pi])
            env.physical_system._ode_solver.set_initial_value(state*scale)
            env.physical_system.mechanical_load._omega = state[0]
            next_obs, _, _, _ = env.step(action)
            next_i_d = next_obs[1]
            next_i_q = next_obs[2]
            next_eps_cos = next_obs[5]
            next_eps_sin = next_obs[6]
            eps_cos = np.cos(state[-1])
            eps_sin = np.sin(state[-1])
            context_content = list()
            context_content.extend(list(state[:-1]))
            context_content.extend([eps_cos,eps_sin])
            context_content.extend(list(action))
            context_content.extend([next_i_d, next_i_q, next_eps_cos, next_eps_sin])
            self.replay_buffer.add(context_content.copy(), motor_id, add_id_buffer=True)
            step=step+1




