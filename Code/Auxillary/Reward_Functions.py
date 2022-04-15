class Scaled_Reward:
    def __init__(self, gamma):
        self.gamma = gamma
    def calc_reward(self, obs, next_obs):
        id = next_obs[1]
        iq = next_obs[2]
        id_ref = obs[4]
        iq_ref = obs[3]
        scale = 1-self.gamma
        if id**2 + iq**2 > 1:
            r = -abs(1- (id**2 + iq**2))*scale/16
        else:
            r = (1- (abs(id-id_ref) + abs(iq-iq_ref))/4)*scale
        return r


