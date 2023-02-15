import os
from utils.utils import test_model
import numpy as np

def mlp_2_layers(amount: int) -> None:
    """Get results of set of runs MLP with 2 layers architecture"""
    mode = 'mlp'
    for i in range(1, amount+1):
        os.system(f"python main.py --mode={mode} --seed={i} --num_layers=2")
    res = test_model(name=mode, path=f'./data/2_layers')

    print_func = lambda k: print(f'\t: {mode} Mean: {round(np.mean(k), 3)} Std: {round(np.std(k), 3)}')
    map(print_func, res)


if __name__ == "__main__":
    amount = 10
    mlp_2_layers(amount)
