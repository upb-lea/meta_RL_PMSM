from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))
from GEM_envs import create_GEM_env
from Buffer import Replay_Buffer
from Rollouts import Rollout
from Reward_Functions import Scaled_Reward

def DESSCA_Test(policy, dessca_samples, num_steps, params, task_id, use_context, context_input_size,
                pre_dessca, max_samples = None):
    current_scale = params['In']
    omega_scale = params['Omegan']

    buffer = Replay_Buffer(dessca_samples.shape[0]*num_steps, task_id+1)
    reward_function = Scaled_Reward(gamma=0.9)
    rollout = Rollout(buffer, reward_function)
    seed = 1
    rewards = list()
    if not isinstance(pre_dessca, type(None)):
        samples = pre_dessca[task_id]
        dessca_states = samples[:, :4]
        dessca_states = dessca_states[:, [2, 0, 1, 3]]
        dessca_actions = samples[:, 4:]
    if max_samples:
        dessca_samples = dessca_samples[:max_samples, :]
    for sample in dessca_samples:
        print(f'Latin Hypercube Sampling Testing at episode {seed}')
        i_sd = sample[0]
        i_sq = sample[1]
        i_sd_ref = sample[3]
        i_sq_ref = sample[4]
        omega = sample[2]
        state_initializer = {
            'i_sd*': i_sd_ref,
            'i_sq*': i_sq_ref,
            'i_sd': i_sd*current_scale,
            'i_sq': i_sq*current_scale,
            'omega': omega*omega_scale,
        }
        env = create_GEM_env(params, True, state_initializer, seed)
        if seed == 1 and not isinstance(pre_dessca, type(None)):
            rollout.initial_rollout(env, task_id, dessca_states, dessca_actions)

        rollout.rollout(env,policy,num_steps,task_id,det=True,single_episode=True, use_context=use_context,
                        context_input_size=context_input_size)
        rewards.extend(rollout.last_rollout_rewards)
        env.close()
        seed += 1
    return rewards, buffer.storage.buffer
