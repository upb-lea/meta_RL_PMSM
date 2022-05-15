from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.absolute().parent))
from Auxillary.MRL import MRL

TRAINING_BATCH_SIZE = 32

#Get userdefined parameters
parser = argparse.ArgumentParser(description='Get Meta-Training Hyperparameters')
parser.add_argument('steps_per_task',
                    help='Number of steps per motor until we draw next')
parser.add_argument('training_steps',
                    help='Number of steps in motors to be taken. Updates will be 5 times less')
parser.add_argument('checkpoint_steps',
                    help='Number of steps after which a checkpoint will be made')

parser.add_argument('-n', '--name', nargs='?',  help='Name of save folder')
parser.add_argument('-s', '--savefolder', nargs='?',  help='Abs path to save folder')
parser.add_argument('-tr', '--trainfile', nargs='?', help='Abs path to .xlsx of training motors')
parser.add_argument('-te', '--testfile', nargs='?', help='Abs path to .xlsx of test motors')
parser.add_argument('-c', '--context', nargs='?', help='Chose the amount of context variables')
args = parser.parse_args()

#Process user input
code_path = Path(__file__).parent.absolute()
train_path = Path(args.trainfile) if args.trainfile is not None else code_path.parent.parent / "MotorDB" / "Training.xlsx"
test_path = Path(args.testfile) if args.testfile is not None else code_path.parent.parent / "MotorDB" / "Test.xlsx"
save_path = Path(args.savefolder) if args.savefolder is not None else code_path.parent.parent / "Save" / "Trainings"
use_context = False if args.context == "None" else True
buffer_size = int(args.training_steps)
context_size = int(args.context)

#Define meta training parameters
policy_params = {
    "state_dim": 8,
    "action_dim": 2,
    "actor_hidden": [100,100],
    "critic_hidden": [100,100],
    "context_hidden": [25,15],
    "actor_lr": 3e-4,
    "critic_lr": 1e-3,
    "exploration_noise": 0.03,
    "exploration_noise_clip": 0.05,
    "gamma": 0.9,
    "tau": 0.005,
    "policy_noise": 0.01,
    "policy_noise_clip": 0.02,
    "policy_freq": 2,
    "context_size": context_size,
    "context_input_size": 1000,
}
#Start traaining
mrl = MRL(train_path,test_path, buffer_size, policy_params, TRAINING_BATCH_SIZE, save_path, use_context=use_context, name=args.name)
mrl.meta_train(int(args.steps_per_task), int(args.training_steps), int(args.checkpoint_steps))
