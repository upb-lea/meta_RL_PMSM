# Meta Reinforcement Learning-Based Current Control of Permanent Magnet Synchronous Motor Drives for a Wide Range of Power Classes
This repository provides the associated engine database and code to the paper Meta Reinforcement Learning-Based Current Control of Permanent Magnet Synchronous Motor Drives for a Wide Range of Power Classes. It aims to allow the reader to reproduce all the results as well as draw inspiration from the code. If you use any sections from the code or the engine database for your publications, please cite this paper as follows:

```
D. Jakobeit, M. Schenke and O. Wallscheid, "Meta-Reinforcement Learning-Based Current Control of Permanent Magnet Synchronous Motor Drives for a Wide Range of Power Classes," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3256424.
```

Or using BibTeX:
```
@ARTICLE{10068250,
  author={Jakobeit, Darius and Schenke, Maximilian and Wallscheid, Oliver},
  journal={IEEE Transactions on Power Electronics}, 
  title={Meta-Reinforcement Learning-Based Current Control of Permanent Magnet Synchronous Motor Drives for a Wide Range of Power Classes}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TPEL.2023.3256424}}

```
## General Setup

<p align="center"> <img src="Supplementary/Meta_Scheme.png" width="600"> </p>

## Engine Database

For this paper, a big number of physical motor drive parameters from a wide range of power classes have been collected. You can find them [here](MotorDB/Complete.xlsx). 

### Parameter Selection for Training and Test

In order to achieve a balanced coverage of the parameter space, a sophisticated set of training and test parameters was selected. This was based on a representation of the parameters that should describe the dynamics of a motor in normalized space. The script can be found and executed in [Dataset_Sampling.py](Code/Data_Selection/Dataset_Sampling.py). Executing it as it is will create the physical parameter sets [Training.xlsx](MotorDB/Training.xlsx) and [Test.xlsx](MotorDB/Test.xlsx) as well as the normalized dynamics parameter set [ODETraining.xlsx](MotorDB/ODETraining.xlsx) and [ODETest.xlsx](MotorDB/ODETest.xlsx) which are used in this work. To derive different sets, the random states in the script have to be altered. Script [Dataset_Visualization.py](Code/Evaluation/Dataset_Visualization.py) can be used to visualize your electrical and ODE parameter distribution.
<p align="center">
  <img src="Supplementary/Ld_Lq.png" width="400" />
  <img src="Supplementary/p1_p4.png" width="400" /> 
</p>

## Commissioning Buffer Sampling

<p align="center"> <img src="Supplementary/dessca_samples.png" width="600"> </p>

This thesis' context variable was designed to be static. Therefore, a comprehensive input had to be selected. This was done by sampling the state/action space using [DESSCA](https://github.com/max-schenke/DESSCA). Executing the script [DESSCA_Sampling.py](Code/Data_Selection/DESSCA_Sampling.py) will result in samples which will be used to gather data for the commissioning buffers from the motors. These samples are saved in the [DESSCA_Samples](Save/DESSCA_Samples/) folder. Additionally, [test episode initializations](Save/DESSCA_Samples/test_routine_samples.npy) are sampled and saved with the execution of the DESSCA_Sampling.py script.

## Starting a Training

The (meta-)training can be started using script [meta_train.py](Code/Routines/meta_train.py). You have to specify the rollout steps per motor drawn, the total number of rollout steps over all motors until training ends and the number of rollout steps after which a checkpoint is created. Using the same values as used in this work this would be

```
meta_train.py 1000 5000000 100000
```
Optionally, using the -tr and -te flags respectively, one can specify a path to the training and test motor parameter sets. Default paths are those of [Training.xlsx](MotorDB/Training.xlsx) and [Test.xlsx](MotorDB/Test.xlsx). The -s flag can be used to specify a save path for the training checkpoints - default is the [Trainings](Save/Trainings) folder. If the -n flag is not specified, the training routine will create a new folder in the [Trainings](Save/Trainings) folder, depicting the datetime of the training start. Alternatively, the -n flag can be used to define a different name. The -c flag can be used to set the amount of context variables used (default is 8). For example a training without a context network could be started using

```
meta_train.py 1000 5000000 100000 -c 0 -n no_context
```
The hyperparameters of the agent have to be changed directly in the script. The default values are the same as used for the models in the paper. For the [MRL](Save/Trainings/MRL) and [RL_AM](Save/Trainings/RL_AM) agents from the in-depth analysis presented in the paper, the respective checkpoints are provided in the repository. We omitted publishing training rewards to not bloat this repository's size.

## Applying the Test Routine

To evaluate a checkpoint's quality, the routine [meta_test.py](Code/Routines/meta_test.py) can be used. You have to specify the folder of the saved agent (its hyperparameters have to be saved in its parent folder). Executing this script will apply the test routine on each motor and return the a value representing its measured control quality. You can optionally use the -b flag to execute only a smaller test routine instead and save the replay buffer for analysis and visualization. The results are saved in the same folder as the tested agent. Another option is to use the -id flag to test the agent only on a specific motor. The pictures depicted below are an example of how the saved buffers can be used to create a visualization of test results, although the data presented here was generated outside of the test routine.
<p align="center">
  <img src="Supplementary/m116_currents.png" width="400" />
  <img src="Supplementary/m67_currents.png" width="400" /> 
</p>

## Evaluate Context Network

When a trained MRL agent is available, script [Generate_Context.py](Code/Evaluation/Generate_Context.py) can be used to generate this agent's context variables for each motor. They are saved in a [Contexts](Save/Trainings/MRL/2800000/Contexts/) folder which lies in the same folder as the evaluated agent. Script [Context_Correlation.py](Code/Evaluation/Generate_Context.py) can then be used to analyze the correlations of these variables to electrical and ODE motor parameters. A resulting picture will also be saved in the context's folder.

<p align="center"> <img src="Supplementary/CorrelationMatrix.png" width="600"> </p>

