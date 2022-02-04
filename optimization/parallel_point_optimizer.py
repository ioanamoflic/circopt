import math
import cirq
import multiprocessing
import os

from optimization.optimize_circuits import CircuitIdentity

NR_PROCS = 3


class ParallelPointOptimizer(cirq.PointOptimizer):

    def __init__(self, only_count=False, parallel = False):
        super().__init__()

        self._manager = None
        self._go_parallel = parallel

        self.only_count = only_count
        self._count = {}
        self._moment_index_qubit = {}


    @property
    def go_parallel(self):
        return self._go_parallel

    @go_parallel.setter
    def go_parallel(self, value):
        self._go_parallel = value
        self._update_process_manager()

    def _update_process_manager(self):
        if self._go_parallel:
            self._manager = multiprocessing.Manager()
            self._count = self._manager.dict()
            self._moment_index_qubit = self._manager.dict()
            self.my_queue = multiprocessing.SimpleQueue()
        else:
            self._manager = None
            self._count = {}
            self._moment_index_qubit = {}
            self.my_queue = None

    @property
    def count(self):
        # print(self._count)
        return sum(self._count.values())


    @property
    def moment_index_qubit(self):
        # print(self._moment_index_qubit)
        ret = []
        for x in self._moment_index_qubit.values():
            ret.extend(x)
        return ret


    def increase_opt_counter(self, ident: CircuitIdentity, index : int, qubit: cirq.Qid):
        pid = os.getpid()

        if pid not in self._count:
            self._count[pid] = 0
        self._count[pid] += 1

        if pid not in self._moment_index_qubit:
            self._moment_index_qubit[pid] = [] if (self._manager is None) else self._manager.list()
        self._moment_index_qubit[pid].append((ident, index, qubit))


    def clean_counters(self):
        self._update_process_manager()


    def local_optimize_circuit(self, circuit, my_queue):
        super().optimize_circuit(circuit)
        if my_queue is not None:
            my_queue.put((os.getpid(), circuit))


    def optimize_circuit(self, circuit: cirq.Circuit):

        if not self.go_parallel:
            self.local_optimize_circuit(circuit, self.my_queue)
            return

        interval_len = int(math.ceil(len(circuit) / NR_PROCS))

        processes = []

        start = 0
        local_circ = [cirq.Circuit()] * NR_PROCS
        pid_to_idx = {}
        for i in range(NR_PROCS):

            local_circ[i]._moments = circuit._moments[start : min(start + interval_len, len(circuit._moments))]
            p = multiprocessing.Process(target=self.local_optimize_circuit,
                                        args=[local_circ[i], self.my_queue])

            start += interval_len

            processes.append(p)
            p.start()

            # Store the pid <-> i relation in order to reconstruct the circuit
            pid_to_idx[p.pid] = i

        for process in processes:
            process.join()

        # print(f"joined...{NR_PROCS}")

        # Order the circuits from the queue
        while not self.my_queue.empty():
            pid_circuit = self.my_queue.get()
            pid = pid_circuit[0]
            idx_subcircuit = pid_to_idx[pid]
            local_circ[idx_subcircuit] = pid_circuit[1]

        # circuit._moments.clear()
        circsum = cirq.Circuit()
        for i in range(NR_PROCS):
            # print(local_circ[i]._moments)
            circsum._moments.extend(local_circ[i]._moments)

        circuit._moments.clear()
        circuit._moments.extend(circsum._moments)