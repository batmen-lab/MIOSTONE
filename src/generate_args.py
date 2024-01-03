import itertools
import os

# Possible values for each argument
seeds = range(10)
datasets = ['ibd200', 'alzbiom', 'asd']
targets = ['type', 'ad', 'stage']
model_types = ['rf', 'mlp', 'taxonn', 'popphycnn', 'miostone']

# Function to validate that the combination of arguments is valid
def validate_args(combination):
    """Validate that the combination of arguments is valid."""
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

# Generate the commands and write them to a file
with open('args.txt', 'w') as f:
    for combination in itertools.product(seeds, datasets, targets, model_types):
        if not validate_args(combination):
            continue
        command = f"--seed {combination[0]} --dataset {combination[1]} --target {combination[2]} --model_type {combination[3]}"
        f.write(command + '\n')
    f.truncate(f.tell() - len(os.linesep)) # Remove the last newline character
    

    