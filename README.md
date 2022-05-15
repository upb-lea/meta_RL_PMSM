# meta_RL_PMSM
This repository provides the associated engine database and code to the paper Meta Reinforcement Learning-Based Current Control of Permanent Magnet Synchronous Motor Drives for a Wide Range of Power Classes. It aims to allow the reader to reproduce all the results as well as draw inspiration from the code. If you use any sections from the code or the engine database for your publications, please cite this paper as follows:

....

## General Setup

<p align="center"> <img src="Supplementary/Meta_Scheme.png" width="600"> </p>

## Engine Database

For this paper a big number of physical motor drive parameters from a wide range of power classes was collected. You can find them [here](MotorDB/Complete.xlsx). 

### Parameter Selection for Training and Test

In order to achieve a balanced coverage of the parameter space, a thought out preselection of training and test parameters was made. This was based on a representation of the parameters that should describe the dynamics of a motor in normalized space. The script can be found and executed in [Dataset_Sampling.py](Code/Data_Selection/Dataset_Sampling.py). Executing it as it is will create the physical parameter sets [training](MotorDB/Training.xlsx) and [test](MotorDB/Test.xlsx) as well as the normalized dynamics parameter set [training](MotorDB/ODETraining.xlsx) and [test](MotorDB/ODETest.xlsx) which are used in this work. To derive different sets, the random states in the script have to be altered.

## Commissioning Buffer Sampling

<p align="center"> <img src="Supplementary/dessca_samples.png" width="600"> </p>

This thesis' context was designed to be static. Therefore, a comprehensive input had to be selected. This was done by sampling the state/action space using [DESSCA](https://github.com/max-schenke/DESSCA). Executing the script [DESSCA_Sampling.py](Code/Data_Selection/DESSCA_Sampling.py) will result in samples which will be used to gather data for the commisioning buffers from the motors. These samples are saved in the [Save](Save/DESSCA_Samples/) folder. Additionally, [test episode initializations](Save/DESSCA_Samples/test_routine_samples.npy) are sampled and saved with the execution of the DESSCA_Sampling.py script.


