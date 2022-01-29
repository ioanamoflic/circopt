import numpy as np

from optimization.reverse_CNOT import ReverseCNOT
from optimization.hadamard_square import HadamardSquare
from optimization.top_left_hadamard import TopLeftHadamard
from optimization.one_H_left_2_right import OneHLeftTwoRight

state_map = {"": 0}
state_map_identity = dict()
action_map = dict()
current_moment = 0
state_counter = dict()

#not used atm
all_configs: np.ndarray
input_depths: np.ndarray
output_depths: np.ndarray
ratios: np.ndarray
times: np.ndarray


counting_optimizers = {
    "onehleft" : OneHLeftTwoRight(only_count=True),
    "toplefth" : TopLeftHadamard(only_count=True),
    "rerversecnot" : ReverseCNOT(only_count=True),
    "hadamardsquare" : HadamardSquare(only_count=True),
}

working_optimizers = {
    "onehleft" : OneHLeftTwoRight(),
    "toplefth" : TopLeftHadamard(),
    "rerversecnot" : ReverseCNOT(),
    "hadamardsquare" : HadamardSquare(),
}

# Global randomness?
import random
random_index = 0
random_moment = 0
def get_random_action(could_apply_on) -> int:
    global action_map
    global random_index
    global random_moment

    random_index = random.randint(0, len(could_apply_on) - 1)
    random_moment = could_apply_on[random_index][1]

    qubit = could_apply_on[random_index][2]
    identity = could_apply_on[random_index][0]

    tuple = (identity, qubit, random.randint(0, 1))
    if tuple not in action_map.keys():
        action_map[tuple] = len(action_map)

    return action_map.get(tuple)


# Timing decorator from stackoverflow
# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

## Parallelisation in non-orthodox way ?
from optimization.parallel_point_optimizer import ParallelPointOptimizer
from cirq import PointOptimizer

def parallelise_optimizers():
    parallel = ParallelPointOptimizer()
    for optimizer in counting_optimizers:
        optimizer.optimize_circuit = parallel.optimize_circuit

    for optimizer in working_optimizers:
        optimizer.optimize_circuit = parallel.optimize_circuit

def deparallelise_optimizers():
    pointopt = PointOptimizer()
    for optimizer in counting_optimizers:
        optimizer.optimize_circuit = pointopt.optimize_circuit

    for optimizer in working_optimizers:
        optimizer.optimize_circuit = pointopt.optimize_circuit