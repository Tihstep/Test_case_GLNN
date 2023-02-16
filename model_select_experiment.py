import os
from utils.utils import test_model
import numpy as np

path = 'C:/Users/Stepan/PycharmProjects/graphless_mlp_interview/data/'

def select_model_exp(amount: int) -> None:
    """Get results of set of runs MLP, GNN, GLNN models with random initialization seed"""

    modes = ['teacher', 'mlp', 'student_mlp']
    results = []

    for mode in modes:
        for i in range(1, amount + 1):
            os.system(f"python main.py --mode={mode} --seed={i}")
            if mode == 'teacher':
                os.system(f"python save_teacher_soft_labels.py --seed={i}")
        results.append(test_model(name=mode, path=path))

    for mode, res in zip(modes, results):
        print_func = lambda k: print(f'\t: {mode} Mean: {round(np.mean(k), 3)} Std: {round(np.std(k), 3)}')
        map(print_func, res)


if __name__ == "__main__":
    amount = 10
    select_model_exp(amount)
