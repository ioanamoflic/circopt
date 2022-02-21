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
              (f.__name__, args, kw, te - ts))
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
