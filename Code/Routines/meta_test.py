from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np
import json
import pickle
import copy
import sys
sys.path.append(str(Path(__file__).parent.absolute().parent))
from Auxillary.MRL import MRL
from Auxillary.Test_Routines import DESSCA_Test

TEST_EPISODE_LENGTH = 50

#Get userdefined parameters
parser = argparse.ArgumentParser(description='Get Meta-Test Information')
parser.add_argument('path_to_model',
                    help='Path to the saved folder where the model lies')
parser.add_argument('-id', '--motorid', nargs='?',  help='ID of the motor, if not specified all motors')
parser.add_argument('-b', '--buffer', nargs='?',  help='Yes when buffer should be saved instead of reward')
args = parser.parse_args()

#Define paths
code_path = Path(__file__).parent.absolute()
dessca_path = code_path.parent.parent / "Save" / "DESSCA_Samples"
path_model = Path(os.path.abspath(args.path_to_model))
path_params = code_path.parent.parent / "MotorDB"

#Load hyperparameters
with open(path_model.parent / 'hyperparams.json', 'r') as hf:
    hyperparameters = json.load(hf)
training_batch_size = hyperparameters['train_batch_size']
use_context = True if hyperparameters["context_size"]>0 else False

#Load DESSCA samples
dessca_samples = np.load(dessca_path / "test_routine_samples.npy")
if use_context:
    dessca_train = np.load(dessca_path / "context_samples_training.npy")
    dessca_test = np.load(dessca_path / "context_samples_test.npy")
    pre_dessca = np.concatenate((dessca_train, dessca_test), 0)
else:
    pre_dessca = None

#Load motor parameter(s)
train_params = pd.read_excel(path_params / "Training.xlsx")
test_params = pd.read_excel(path_params / "Test.xlsx")
all_params = pd.concat((train_params, test_params), ignore_index = True)
if args.motorid:
    all_params = all_params.iloc[int(args.motorid)].to_frame().T

#Parameters when -b flag is specified
return_buffer = True if args.buffer == "Yes" else False
max_samples = 500 if args.buffer == "Yes" else None

#Load MRL agent
mrl = MRL(path_params / "Training.xlsx",path_params / "Test.xlsx",12345, hyperparameters, training_batch_size, path_model, use_context)
mrl.load(target=True)

#Conduct test routine on motor id(s)
rewards_dict = dict()
buffers_dict = dict()
for index, row in all_params.iterrows():
    print(index, row)
    rewards, buffer = DESSCA_Test(mrl.td3_policy, dessca_samples, TEST_EPISODE_LENGTH, row, index, use_context,
                          hyperparameters["context_input_size"], pre_dessca, max_samples)

    rewards_dict[str(index)] = np.sum(rewards)/int(1e6)
    if return_buffer:
        buffers_dict[str(index)] = buffer

#Save either rewards or buffer(s)
if not return_buffer:
    if args.motorid:
        save_path = path_model / f'evaluation_{args.motorid}.json'
    else:
        save_path = path_model / f'evaluation.json'
    with open(save_path, 'w') as savefile:
        json.dump(rewards_dict, savefile)
else:
    if args.motorid:
        save_path = path_model / f'buffer_{args.motorid}'
    else:
        save_path = path_model / f'buffer'
    with open(save_path, 'wb') as savefile:
        pickle.dump(buffers_dict, savefile)

