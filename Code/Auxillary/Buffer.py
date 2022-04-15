from numpy.random import default_rng
import pickle
from random import sample

#Modified from https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
class ReplayMemory:
    def __init__(self, max_size, max_ids):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.curr_pos = 0
        self.size = 0
        self.id_buffers = [None] * max_ids
        for i in range(max_ids):
            self.id_buffers[i] = ReplayMemory(1000,0)

    def append(self, obj, motor_id=None, ):
        if motor_id is not None:
            self.id_buffers[motor_id].append(obj)
        else:
            self.buffer[self.curr_pos] = obj
            self.size = min(self.size + 1, self.max_size)
            self.curr_pos = (self.curr_pos + 1) % self.max_size

    def sample(self, batch_size, motor_id=None):
        if motor_id is not None:
            samples = self.id_buffers[motor_id].sample(batch_size)
        else:
            indices = sample(range(self.size), batch_size)
            samples = [self.buffer[index] for index in indices]
        return samples


class Replay_Buffer:
    def __init__(self, max_len, max_ids):
        self.storage = ReplayMemory(max_size=max_len, max_ids=max_ids)
        self.rng = default_rng()

    def add(self, data, motor_id, add_id_buffer=False):
        if not add_id_buffer:
            self.storage.append([data,motor_id], None)
        else:
            self.storage.append(data, motor_id)

    def save_storage(self, path):
        with open(path / 'storage', 'wb') as storage_file:
            pickle.dump(self.storage, storage_file)

    def load_storage(self, path):
        with open(path / 'storage', 'rb') as storage_file:
            self.storage = pickle.load(storage_file)

    def sample(self, batch_size, motor_id=None):
        observations, next_observations, actions, rewards, motor_ids = [], [], [], [], []
        data_samples = self.storage.sample(batch_size,motor_id)
        if motor_id == None:
            for data_sample in data_samples:
                motor_id = data_sample[1]
                values = data_sample[0]
                observations.append(values[0])
                next_observations.append(values[1])
                actions.append(values[2])
                rewards.append(values[3])
                motor_ids.append(motor_id)
            return observations, next_observations, actions, rewards, motor_ids
        else:
            return data_samples

