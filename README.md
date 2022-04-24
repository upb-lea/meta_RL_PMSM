# meta_RL_PMSM
This repository provides the associated engine database and code to the paper Meta Reinforcement Learning-Based Current Control of Permanent Magnet Synchronous Motor Drives for a Wide Range of Power Classes. It aims to allow the reader to reproduce all the results as well as draw inspiration from the code. If you use any sections from the code or the engine database for your publications, please cite this paper as follows:

....

## Engine Database

For this paper a big number of physical motor drive parameters from a wide range of power classes was collected. You can find them in the folder _MotorDB_ in the file _Complete.xlsx_. 

### Parameter Selection for Training and Test

In order to achieve a balanced coverage of the parameter space, a thought out preselection of training and test parameters was made. This was based on a representation of the parameters that should describe the dynamics of a motor in normalized space. The script can be found and executed in _Code/Data_Selection/Dataset_Sampling.py_. Executing it as it is will create the physical parameter sets _Training.xlsx_ and _Test.xlsx_ as well as the normalized dynamics parameter set _ODETraining.xlsl_ amd _ODETest.xlsx_ which are used in this work. To derive different sets, the random states in _Dataset_Sampling.py_ have to be altered.


