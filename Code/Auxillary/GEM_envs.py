import pandas as pd
import gym_electric_motor as gem
from gym import Wrapper, ObservationWrapper
from gym_electric_motor.reference_generators import \
    MultipleReferenceGenerator,\
    SubepisodedReferenceGenerator
import numpy as np
from gym_electric_motor.visualization import ConsolePrinter,MotorDashboard
from gym.wrappers import FlattenObservation, TimeLimit
from gym.spaces import Box
from numpy.random import default_rng


class SqdCurrentMonitor(gem.constraints.Constraint):
    I_SD_IDX = 0
    I_SQ_IDX = 0

    def set_modules(self, physical_system):
        self.I_SD_IDX = physical_system.state_positions['i_sd']
        self.I_SQ_IDX = physical_system.state_positions['i_sq']

    def __call__(self, state):
        """This function returns a "violation degree" within [0,1].

        This exemplary constraint returns 0 if i_sd is 0 or below.
        It returns a value between 0 and 1 if i_sd is at most half of the generally allowed current limit.
        If i_sd exceeds 0.5 of the current limit, the violation degree is 1.0 and the episode will be terminated.

        The violation degree between 0 and 1 directly affects the reward, but does not end an episode.
        It models an "undesired behavior" the agent shall learn to avoid.
        """
        sqd_currents = state[self.I_SD_IDX]** 2 + state[self.I_SQ_IDX] ** 2
        return sqd_currents > 1


class WienerProcessGenerator(SubepisodedReferenceGenerator):
    """
    Reference Generator that generates a reference for one state by a Wiener Process with the changing parameter sigma
    and mean = 0.
    """

    _current_sigma = 0

    def __init__(self, sigma_range=(1e-3, 1e-1), reference_value=0.0, seed=None, ref_gen=None, test=False, *_, **kwargs):
        """
        Args:
            sigma_range(Tuple(float,float)): Lower and Upper limit for the sigma-parameter of the WienerProcess.
            kwargs: Further arguments to pass to SubepisodedReferenceGenerator
        """
        super().__init__(**kwargs)
        self.rng = default_rng()
        self._reference_value=0
        self._sigma_range = sigma_range
        self.seed = seed
        self.ref_gen = ref_gen
        self.winner = False #Who has to change his val
        self.test = test
        self._initial_value = reference_value

    def _reset_reference(self):
        self.winner = ~self.winner
        upper_limit = 0 if self._reference_state == 'i_sd' else 1
        self._current_sigma = 10 ** self._get_current_value(np.log10(self._sigma_range))
        if self.seed:
            np.random.seed(self.seed)
        random_values = np.random.normal(0, self._current_sigma, self._current_episode_length)
        self._reference = np.zeros_like(random_values)
        if self.test:
           add = 0.0
        else:
            add = self.rng.uniform(-1,upper_limit)
        reference_value = self._initial_value + add
        for i in range(self._current_episode_length):
            if self.test:
                sub =(reference_value - self._initial_value)*0.3
            else:
                sub = 0.0
            reference_value += random_values[i] - sub
            if reference_value > upper_limit:
                reference_value = upper_limit
            if reference_value < -1:
                reference_value = -1
            if self.ref_gen is not None:
                if reference_value**2 + self.ref_gen._reference[i]**2 > 1:
                    if self.winner:
                        reference_value = -1*np.sqrt(1-self.ref_gen._reference[i]**2)
                    else:
                        if self.ref_gen._reference[i] < 0:
                            self.ref_gen._reference[i] = -1*np.sqrt(1-reference_value**2)
                        else:
                            self.ref_gen._reference[i] = np.sqrt(1-reference_value**2)
            self._reference[i] = reference_value


class FeatureWrapper(ObservationWrapper):
    """
    Wrapper class which wraps the environment to change its observation. Serves
    the purpose to improve the agent's learning speed.

    It changes epsilon to cos(epsilon) and sin(epsilon). This serves the purpose
    to have the angles -pi and pi close to each other numerically without losing
    any information on the angle.

    Additionally, this wrapper adds a new observation i_sd**2 + i_sq**2. This should
    help the agent to easier detect incoming limit violations.
    """

    def __init__(self, env):
        """
        Changes the observation space to fit the new features

        Args:
            env(GEM env): GEM environment to wrap
            epsilon_idx(integer): Epsilon's index in the observation array
            i_sd_idx(integer): I_sd's index in the observation array
            i_sq_idx(integer): I_sq's index in the observation array
        """
        super(FeatureWrapper, self).__init__(env)
        self.EPSILON_IDX = env.physical_system.state_names.index('epsilon')
        self.I_SQ_IDX = env.physical_system.state_names.index('i_sq')
        self.I_SD_IDX = env.physical_system.state_names.index('i_sd')
        U_A_IDX = env.physical_system.state_names.index('u_a')
        U_B_IDX = env.physical_system.state_names.index('u_b')
        U_C_IDX = env.physical_system.state_names.index('u_c')
        I_A_IDX = env.physical_system.state_names.index('i_a')
        I_B_IDX = env.physical_system.state_names.index('i_b')
        I_C_IDX = env.physical_system.state_names.index('i_c')
        U_SUP_IDX = env.physical_system.state_names.index('u_sup')
        U_SD_IDX = env.physical_system.state_names.index('u_sd')
        U_SQ_IDX = env.physical_system.state_names.index('u_sq')
        T_IDX = env.physical_system.state_names.index('torque')
        self.delete_arr = [self.EPSILON_IDX, I_A_IDX, I_B_IDX, I_C_IDX,
                           U_A_IDX, U_B_IDX, U_C_IDX, U_SUP_IDX, T_IDX,
                           U_SD_IDX, U_SQ_IDX]
        new_low = np.delete(self.env.observation_space.low, self.delete_arr)
        new_low = np.concatenate((new_low, np.array([-1.]), np.array([-1.]), np.array([0.])))

        new_high = np.delete(self.env.observation_space.high, self.delete_arr)
        new_high = np.concatenate((new_high, np.array([1.]), np.array([1.]), np.array([1.])))
        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        """
        Gets called at each return of an observation. Adds the new features to the
        observation and removes original epsilon.

        """
        cos_eps = np.cos(observation[self.EPSILON_IDX] * np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX] * np.pi)
        currents_squared = observation[self.I_SQ_IDX] ** 2 + observation[self.I_SD_IDX] ** 2
        observation = np.delete(observation, self.delete_arr)
        observation = np.concatenate((observation, np.array([cos_eps, sin_eps, currents_squared])))
        return observation


def create_GEM_env(params, test=False, state_initializer=None, seed=None):

    MAX_EP_STEPS = 500

    motor_parameter = dict(p=params['p'],  # [p] = 1, nb of pole pairs
                       r_s=params['Rs'],  # [r_s] = Ohm, stator resistance
                       l_d=params['Ld'],  # [l_d] = H, d-axis inductance
                       l_q=params['Lq'],  # [l_q] = H, q-axis inductance
                       psi_p=params['Psip'],  # [psi_p] = Vs, magnetic flux of the permanent magnet
                       )

    #This reward function is unused in the final application
    reward_function = gem.reward_functions.WeightedSumOfErrors(
        observed_states=['i_sq', 'i_sd'],
        reward_weights={'i_sq': 1, 'i_sd': 1},
        gamma=0.8,
        reward_power=1)

    nominal_values=dict(omega=params['Omegan'],
                        i=params['In'],
                        u=params['UDC']
                        )
    limit_values=dict(omega=params['Omegan'],
                        i=1.5*params['In'],
                        u=params['UDC']
                        )

    episode_lengths = (50,50) if test else (MAX_EP_STEPS,MAX_EP_STEPS)
    ref_q = state_initializer['i_sq*'] if state_initializer is not None else 0.0
    ref_d = state_initializer['i_sd*'] if state_initializer is not None else 0.0

    q_generator = WienerProcessGenerator(reference_state='i_sq', reference_value=ref_q,
                                         episode_lengths=episode_lengths, seed=seed, test=test)
    d_generator = WienerProcessGenerator(reference_state='i_sd', reference_value=ref_d, episode_lengths=episode_lengths,
                                         ref_gen=q_generator, seed=seed + 500 if seed is not None else seed, test=test)
    rg = MultipleReferenceGenerator([q_generator, d_generator])
    if not test:
        motor_initializer = {'random_init': 'uniform', 'interval': [[-params['In'], params['In']],
                                                                    [-params['In'], params['In']], [-np.pi, np.pi]]}
        load_initializer = {'random_init': 'uniform', }
    else:
        motor_initializer = {'states': {'i_sd': state_initializer['i_sd'], 'i_sq': state_initializer['i_sq'], }, }
        load_initializer = {'states': {'omega': state_initializer['omega'], }, }


    visualization = ConsolePrinter()
    env = gem.make(  # define a PMSM with continuous action space
        "PMSMCont-v1",
        # visualize the results
        visualization=visualization,
        control_space = 'dq',
        # parameterize the PMSM and update limitations
        motor_parameter=motor_parameter,
        limit_values=limit_values, nominal_values=nominal_values,
        # define the random initialisation for load and motor
        load='ConstSpeedLoad',
        load_initializer=load_initializer,
        motor_initializer=motor_initializer,
        reward_function=reward_function,
        constraints=[SqdCurrentMonitor()],
        # define the duration of one sampling step
        tau=1e-4, u_sup=params['UDC'],
        # turn off terminations via limit violation, parameterize the rew-fct
        reference_generator=rg, ode_solver='euler',
    )
    env = TimeLimit(FeatureWrapper(FlattenObservation(env)), MAX_EP_STEPS)
    return env



class Env_Generator:
    def __init__(self, train_path, test_path):
        self.train_params = pd.read_excel(train_path)
        self.test_params = pd.read_excel(test_path)

    def get_sizes(self):
        sizes = {
            'Train': self.train_params.shape[0],
            'Test': self.test_params.shape[0],
        }
        return sizes

    def get_env(self, motor_id, param_set):
        if param_set == 'Train':
            params = self.train_params.iloc[motor_id]
        elif param_set == 'Test':
            params = self.test_params.iloc[motor_id]
        print(params)
        GEM_env = create_GEM_env(params)
        return GEM_env