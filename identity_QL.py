from time import sleep

import gym
import cirq

from RL.circuit_env_identities import CircuitEnvIdent
from quantify.mathematics.carry_ripple_4t_adder import CarryRipple4TAdder
from RL.q_learning import QAgent
import routing.routing_multiple as rm

import global_stuff as g
from circopt_utils import get_all_possible_identities
from examples.bernstein import bernstein_vazirani


def run():
    bits: int = 5

    # starting_circuit: cirq.Circuit = CarryRipple4TAdder(bits).circuit
    starting_circuit = bernstein_vazirani(nr_bits=6, secret="101100")
    print(starting_circuit)

    circ_dec = rm.RoutingMultiple(starting_circuit, no_decomp_sets=10, nr_bits=bits)
    circ_dec.get_random_decomposition_configuration()

    nr_qlearn_trials = 1
    for i in range(nr_qlearn_trials):
        conf = circ_dec.configurations.pop()
        decomposed_circuit = circ_dec.decompose_toffolis_in_circuit(conf)
        possible_identities = get_all_possible_identities(decomposed_circuit)
        g.state_map_identity = {}
        g.state_counter = {}

        env = CircuitEnvIdent(decomposed_circuit, could_apply_on=possible_identities)
        agent = QAgent(env, n_ep=3000, max_iter=2000, lr=0.2, gamma=0.99)

        # TODO: Alexandru De aici in jos e interesant
        agent.train()
        agent.show_evolution(conf)

        # to see prints
        sleep(5)


if __name__ == '__main__':
    run()
