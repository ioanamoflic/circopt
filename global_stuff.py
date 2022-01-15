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




