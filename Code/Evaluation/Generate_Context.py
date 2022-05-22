from pathlib import Path
import sys
sys.path.append(str((Path(__file__).resolve().parent.parent / "Auxillary" ).absolute()))
from MRL import MRL
from pathlib import Path
import argparse
import json
import os
import pandas as pd
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Get information for the context generation')
parser.add_argument('path_to_model',
                    help='Path to the folder where the saved model lies')
args = parser.parse_args()

code_path = Path(__file__).parent.absolute()

path_train = code_path.parent.parent / "MotorDB" / "Training.xlsx"
path_test = code_path.parent.parent / "MotorDB" / "Test.xlsx"
path_model = Path(os.path.abspath(args.path_to_model))

path_dessca = code_path.parent.parent / "Save" / "DESSCA_Samples"
dessca_train = np.load(path_dessca / "context_samples_training.npy")
dessca_test = np.load(path_dessca / "context_samples_test.npy")
dessca = np.concatenate((dessca_train, dessca_test), axis = 0)

with open(path_model.parent / 'hyperparams.json', 'r') as hf:
    hyperparameters = json.load(hf)

training_batch_size = hyperparameters['train_batch_size']
mrl = MRL(path_train,path_test,50000, hyperparameters, training_batch_size, path_model)
mrl.load()

save_path = path_model / "Contexts"
save_path.mkdir(parents=True, exist_ok=True)

train_contexts = [None] * (mrl.num_train)
for task_id in range(mrl.num_train):
    dessca_states = dessca[task_id, :, :4]
    dessca_states = dessca_states[:, [2, 0, 1, 3]]
    dessca_actions = dessca[task_id, :, 4:]
    env = mrl.env_generator.get_env(task_id, "Train")
    mrl.train_rollout.initial_rollout(env, task_id, dessca_states, dessca_actions)
    transitions = mrl.train_buffer.sample(1000, task_id)
    transitions = torch.FloatTensor(transitions)
    transitions = torch.unsqueeze(transitions, 0)
    context = mrl.td3_policy.context(transitions.float())
    train_contexts[task_id] = context[0].detach().numpy()
    print(context)

columns = [f"context{i}" for i in range(8)]
df = pd.DataFrame(train_contexts, columns = columns)
df.to_excel(save_path / "TrainContexts.xlsx", index=False)

test_contexts = [None] * (mrl.num_test)
for task_id in range(mrl.num_test):
    dessca_states = dessca[task_id + mrl.num_train, :, :4]
    dessca_states = dessca_states[:, [2, 0, 1, 3]]
    dessca_actions = dessca[task_id + mrl.num_train, :, 4:]
    env = mrl.env_generator.get_env(task_id, "Test")
    mrl.train_rollout.initial_rollout(env, task_id, dessca_states, dessca_actions)
    transitions = mrl.train_buffer.sample(1000, task_id)
    transitions = torch.FloatTensor(transitions)
    transitions = torch.unsqueeze(transitions, 0)
    context = mrl.td3_policy.context(transitions.float())
    test_contexts[task_id] = context[0].detach().numpy()
    print(context)

df = pd.DataFrame(test_contexts, columns = columns)
df.to_excel(save_path / "TestContexts.xlsx", index=False)
