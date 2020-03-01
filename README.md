# Sage Health Coronavirus Deep Learning Competition #

Please refer to the jupyter notebook [rl_trainer.ipynb](https://github.com/flyyufelix/nCoV_DRL_Competition/blob/master/rl_trainer.ipynb) for detailed run through of the entire pipeline. 

## Method Overview: ##

Our approach is an adaptation of  [Deep Reinforcement Learning for de-novo Drug Design](https://github.com/isayev/ReLeaSE). The idea is to first train a generative Recurrent Neural Network (Stack-RNN) on 1.5 million structures on ChEMBL22, then fine-tune the generative RNN with Reinforcement Learning using the binding affinity score of the generated ligand with  [6LU7 3C Protease](https://www.rcsb.org/structure/6LU7)  as reward.

We use  [AutoDock Vina](http://vina.scripps.edu/)  to perform ligand-protein binding simulation.

## Run Instruction ##

The code for reproducing the results in the `rl_trainer.ipynb` jupyter notebook can be found in `run_rl.py`. You can run the code via Docker container:

```
$ sh run_docker.sh
```

This will build and spin up a Docker container where `run_rl.py` will be run automatically.

Alternatively, you can manually install all the dependencies:

In order to get started you will need to install:

-   [Pytorch 1.1.0](https://pytorch.org/)
-   [RDKit](https://www.rdkit.org/docs/Install.html)
-   [Babel]([http://openbabel.org/wiki/Main_Page](http://openbabel.org/wiki/Main_Page)
-   [AutoDock Vina]([http://vina.scripps.edu/](http://vina.scripps.edu/)

All the python dependencies are specified in `requirements.txt`. You can install it via:

```
$ pip install -r requirements.txt
```

GPU is not required to run the code. The speed bottleneck is on AutoDock Vina docking simulation. 
