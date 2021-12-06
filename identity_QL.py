from time import sleep

import gym
import cirq

from RL.circuit_env_identities import CircuitEnvIdent
from quantify.mathematics.carry_ripple_4t_adder import CarryRipple4TAdder
from RL.q_learning import QAgent
import routing.routing_multiple as rm

import config as c

from circopt_utils import get_all_possible_identities
from examples.bernstein import bernstein_vazirani

def run():
    bits: int = 5

    # starting_circuit: cirq.Circuit = CarryRipple4TAdder(bits).circuit
    starting_circuit = bernstein_vazirani(nr_bits=2, secret="11")

    circ_dec = rm.RoutingMultiple(starting_circuit, no_decomp_sets=10, nr_bits=bits)
    circ_dec.get_random_decomposition_configuration()

    nr_qlearn_trials = 1
    for i in range(nr_qlearn_trials):

        #
        conf = circ_dec.configurations.pop()
        decomposed_circuit = circ_dec.decompose_toffolis_in_circuit(conf)
        possible_identities = get_all_possible_identities(decomposed_circuit)
        c.state_map_identity = {}

        env = CircuitEnvIdent(decomposed_circuit, could_apply_on=possible_identities)
        agent = QAgent(env, n_ep=10, max_iter=len(possible_identities), lr=0.05, gamma=0.85)
        agent.train()
        agent.show_evolution(conf)

        # to see prints
        sleep(5)


if __name__ == '__main__':
    run()
