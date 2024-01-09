import itertools
import os

# Possible values for each argument
seeds = range(10)
datasets = ['ibd200', 'alzbiom', 'asd', 'hmp2']
targets = ['type', 'ad', 'stage']
model_types = ['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone']
tl_epochs = [0, 25, 50, 75, 100]

# Function to validate that the combination of arguments for training is valid
def validate_train_args(combination):
    seed, dataset, target, model_type = combination
    if dataset == 'ibd200' and target != 'type':
        return False
    elif dataset == 'alzbiom' and target != 'ad':
        return False
    elif dataset == 'asd' and target != 'stage':
        return False
    elif dataset == 'gd' and target != 'cohort':
        return False
    elif dataset == 'agp' and target != 'sex':
        return False
    elif dataset == 'hmp1' and target != 'sex':
        return False
    elif dataset == 'hmp2' and target != 'type':
        return False
    return True

# Function to validate that the combination of arguments for transfer learning is valid
def validate_transfer_learning_args(combination):
    seed, model_type, tl_epochs = combination
    if model_type == 'rf' and tl_epochs != 0:
        return False
    return True

# Generate the commands and write them to a file
with open('train_args.txt', 'w') as f:
    for combination in itertools.product(seeds, datasets, targets, model_types):
        if not validate_train_args(combination):
            continue
        command = f"--seed {combination[0]} --dataset {combination[1]} --target {combination[2]} --model_type {combination[3]}"
        f.write(command + '\n')

with open('transfer_learning_args.txt', 'w') as f:
    for combination in itertools.product(seeds, model_types, tl_epochs):
        if not validate_transfer_learning_args(combination):
            continue
        command = f"--seed {combination[0]} --dataset ibd200 --pretrain_dataset hmp2 --target type --model_type {combination[1]} --tl_epochs {combination[2]}"
        f.write(command + '\n')
    

    