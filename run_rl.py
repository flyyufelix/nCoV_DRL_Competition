import sys
import os
sys.path.append('./release/')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm, trange
import pickle
import subprocess
from rdkit import Chem, DataStructs
from stackRNN import StackAugmentedRNN
from data import GeneratorData 
from utils import canonical_smiles

import matplotlib.pyplot as plt
import seaborn as sns

use_cuda = torch.cuda.is_available()

gen_data_path = './data/chembl_22_clean_1576904_sorted_std_final.smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

print("Parsing Data...")

gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', 
                         cols_to_read=[0], keep_header=True, tokens=tokens)

# Params for RNN Generator
hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.001
optimizer_instance = torch.optim.Adadelta

from reinforcement import Reinforcement

my_generator_max = StackAugmentedRNN(input_size=gen_data.n_characters, 
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters, 
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth, 
                                     use_cuda=use_cuda, 
                                     optimizer_instance=optimizer_instance, lr=lr)

model_path = './checkpoints/generator/checkpoint_biggest_rnn'

my_generator_max.load_model(model_path)

# Params for RL
n_to_generate = 200
n_policy_replay = 10
n_policy = 30
n_iterations = 60

def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def get_docking_reward(smile):
    """
    Reward function for RL. Use AutoDock Vina to simulate binding affinity of our ligand with receptor protein. Use binding affinity as reward to train the RL agent 
    """

	#vina --receptor receptor.pdbqt --ligand ligand.pdbqt --log dock.log --exhaustiveness 3 --center_x -10 --center_y 10 --center_z 70 --size_x 10 --size_y 15 --size_z 15
    vina_cmd = ['./vina','--receptor','receptor.pdbqt','--ligand','ligand.pdbqt','--log','dock.log','--exhaustiveness','3','--center_x','-10','--center_y','10','--center_z','70','--size_x','10','--size_y','15','--size_z','15']

    # Check SMILE validity
    binding_score = 0 # Penalty for invalid SMILES
    if Chem.MolFromSmiles(smile) is None:
        print("Invalid SMILE: " + str(smile))
        return binding_score
    else:
        # Transform SMILES to pdbqt
        print("Valid SMILE: " + str(smile))
        if os.path.exists('ligand.smi'):
            os.remove('ligand.smi')
        if os.path.exists('ligand.pdbqt'):
            os.remove('ligand.pdbqt')
        with open('ligand.smi', 'w') as f:
            f.write(smile)
        babel_cmd = ['babel','ligand.smi','ligand.pdbqt','--gen3D']
        out = subprocess.Popen(babel_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out.communicate()
        out = subprocess.Popen(vina_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out.communicate()
        with open('dock.log', 'r') as f:
            for line in f:
                if "  1  " in line:
                    binding_score = line.split('      ')[1].strip()
        try:
            binding_score = -float(binding_score)
        except:
            pass

    reward = 0.0
    if binding_score > 6.0:
        reward = 30.0
    elif binding_score > 5.5:
        reward = 20.0
    elif binding_score > 5.0:
        reward = 10.0
    elif binding_score > 4.5:
        reward = 5.0
    elif binding_score > 0.0:
        reward = 1.0

    print("Binding Score: " + str(binding_score) + " Reward: " + str(reward))
    return reward

RL_agent = Reinforcement(my_generator_max, get_docking_reward)

rewards = []
rl_losses = []

for i in range(n_iterations):
    for j in trange(n_policy, desc='Policy gradient...'):
        cur_reward, cur_loss = RL_agent.policy_gradient(gen_data)
        rewards.append(simple_moving_average(rewards, cur_reward)) 
        print("Rewards: " + str(rewards))
        rl_losses.append(simple_moving_average(rl_losses, cur_loss))
        print("Losses: " + str(rl_losses))

