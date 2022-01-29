import math
import cirq
import multiprocessing

NR_PROCS = 3

class ParallelPointOptimizer(cirq.PointOptimizer):

    def __init__(self, go_parallel = False):
        super().__init__()
        self.go_parallel = go_parallel

    def optimize_circuit(self, circuit: cirq.Circuit):

        if not self.go_parallel:
            super().optimize_circuit(circuit)
            return

        interval_len = int(math.ceil(len(circuit) / NR_PROCS))

        processes = []

        start = 0
        local_circ = [cirq.Circuit()] * NR_PROCS
        for i in range(NR_PROCS):
            local_circ[i]._moments = circuit._moments[start : min(start + interval_len, len(circuit._moments))]
            p = multiprocessing.Process(target=super().optimize_circuit,
                                        args=[local_circ[i], ])

            start += interval_len

            processes.append(p)
            p.start()

        for process in processes:
            process.join()

        # print(f"joined...{NR_PROCS}")

        circuit._moments.clear()
        for i in range(NR_PROCS):
            circuit._moments.extend(local_circ[i]._moments)