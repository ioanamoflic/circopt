import quantify.mathematics as mathematics
import cirq.contrib.routing as ccr
from routing.routing_multiple import RoutingMultiple
import csv


def main():
    device_graph = ccr.get_grid_device_graph(20, 20)

    header = ['nrbits', 'toffoli_desc_conf', 'nr_qubits_output', 'depth_input', 'depth_output', 'process_time']
    with open('data.csv', 'w', encoding='UTF8', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for nr_bits in range(4, 15):
            circuit = mathematics.CarryRipple8TAdder(nr_bits, False).circuit
            csv_lines = RoutingMultiple(circuit, device_graph, 100, nr_bits).route_circuit_for_multiple_configurations()
            writer.writerows(csv_lines)
            print("lines: ", csv_lines)


if __name__ == '__main__':
    main()
