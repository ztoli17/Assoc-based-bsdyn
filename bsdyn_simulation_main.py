import random
import os
import pickle
from typing import List
import numpy as np
import matplotlib

matplotlib.use('Agg')
import igraph as ig
from numba import njit, int64, float64, boolean, types, typed
from numba.experimental import jitclass

spec_1 = [
    ('bs_size', int64),
    ('dissonance_penalty', float64),
    ('const_negative_nodes', int64[:]),
    ('const_positive_nodes', int64[:]),
    ('temperature', float64),
    ('communication_memory', float64[:]),
    ('coherency_level', float64),
    ('attitude_list', float64[:]),
    ('assoc_mtx', float64[:, :]),
    ('assoc_sum', float64),
    ('use_clones', boolean),
    ('comm_counter', float64[:, :]),
    ('max_stab', int64),
    ('attitude_change_ampl', float64)
]


@jitclass(spec_1)
class Agent:
    """
    Class representing individual agents. Includes
    """

    def __init__(self, bs_size: int, dissonance_penalty: int, const_negative_nodes: List[int],
                 const_positive_nodes: List[int], temperature: float,
                 max_stab: int, attitude_change_ampl: float):
        self.bs_size = bs_size
        self.const_negative_nodes = const_negative_nodes
        self.const_positive_nodes = const_positive_nodes
        self.coherency_level = 0
        self.assoc_mtx = np.empty((self.bs_size, self.bs_size))
        self.communication_memory = np.empty(self.bs_size * 2)
        self.temperature = temperature
        self.max_stab = max_stab
        self.dissonance_penalty = dissonance_penalty
        self.comm_counter = np.zeros((self.bs_size, self.bs_size))
        self.attitude_change_ampl = attitude_change_ampl

    def bs_gen(self, starter, starting_assoc_mtx=np.empty((1, 1)), starting_attitudes=np.empty(1)):
        """
        Function generating the belief system of the agent, either from scratch or from given starting values.
        """
        if starter:
            assoc_mtx = np.random.uniform(0.0, 1.0, size=(self.bs_size, self.bs_size))
            np.fill_diagonal(assoc_mtx, 0)
            for i in range(self.bs_size):
                for j in range(i, self.bs_size):
                    assoc_mtx[j, i] = assoc_mtx[i, j]
            attitude_list = np.random.uniform(-1.0, 1.0, size=self.bs_size)
            for neg in self.const_negative_nodes:
                attitude_list[neg] = -1
            for pos in self.const_positive_nodes:
                attitude_list[pos] = 1
        else:
            assoc_mtx = starting_assoc_mtx.copy()
            attitude_list = starting_attitudes
        self.assoc_mtx = assoc_mtx
        self.assoc_sum = assoc_mtx.sum()
        self.attitude_list = attitude_list
        self.coherency_level = self.calculate_coherency_level()
        if starter:
            for i in range(10 * self.bs_size):
                self.bs_stab(-1, 1 / (1 + i) * self.temperature)

    def calculate_coherency_level(self):
        coherency_level = 0
        for i in range(self.bs_size):
            for j in range(i + 1, self.bs_size):
                coherency = self.assoc_mtx[i, j] * self.attitude_list[i] * self.attitude_list[j]
                if coherency < 0:
                    coherency_level += self.dissonance_penalty * coherency
                else:
                    coherency_level += coherency
        coherency_level = coherency_level / (self.bs_size * (self.bs_size - 1) / 2)
        return coherency_level

    def bs_stab(self, concept_id: int, temperature):
        """
        Belief system stabilization process of the agent
        """
        if concept_id == -1:
            starting_coherency = self.coherency_level
            starting_attitudes = self.attitude_list.copy()
            for i in range(self.max_stab):
                concept_random = np.random.choice(
                    np.arange(self.bs_size - len(self.const_negative_nodes) - len(self.const_positive_nodes)))
                if concept_random not in self.const_positive_nodes and concept_random not in self.const_negative_nodes:
                    random_push = random.uniform(-1, 1)
                    if random_push > 0 and self.attitude_list[concept_random] > 0:
                        attitude_valt_ampl_used = 1 - self.attitude_list[concept_random]
                    elif random_push < 0 and self.attitude_list[concept_random] < 0:
                        attitude_valt_ampl_used = abs(-1 - self.attitude_list[concept_random])
                    else:
                        attitude_valt_ampl_used = self.attitude_change_ampl
                    self.attitude_list[concept_random] += (random_push * attitude_valt_ampl_used)
                    if self.attitude_list[concept_random] < -1:
                        self.attitude_list[concept_random] = -1
                    elif self.attitude_list[concept_random] > 1:
                        self.attitude_list[concept_random] = 1
                    self.coherency_level = self.calculate_coherency_level()
                    coherence_diff = self.coherency_level - starting_coherency
                    if np.exp(coherence_diff / temperature * (i + 1)) > np.random.random():
                        starting_attitudes = self.attitude_list.copy()
                        starting_coherency = self.coherency_level
                    else:
                        self.attitude_list = starting_attitudes.copy()
                        self.coherency_level = starting_coherency

        elif concept_id not in self.const_positive_nodes and concept_id not in self.const_negative_nodes:
            starting_coherency = self.coherency_level
            starting_attitudes = self.attitude_list.copy()
            for i in range(self.max_stab):
                random_push = random.uniform(-1, 1)
                if random_push > 0 and self.attitude_list[concept_id] > 0:
                    attitude_valt_ampl_used = 1 - self.attitude_list[concept_id]
                elif random_push < 0 and self.attitude_list[concept_id] < 0:
                    attitude_valt_ampl_used = abs(-1 - self.attitude_list[concept_id])
                else:
                    attitude_valt_ampl_used = self.attitude_change_ampl
                self.attitude_list[concept_id] += (random_push * attitude_valt_ampl_used)
                if self.attitude_list[concept_id] < -1:
                    self.attitude_list[concept_id] = -1
                elif self.attitude_list[concept_id] > 1:
                    self.attitude_list[concept_id] = 1
                self.coherency_level = self.calculate_coherency_level()  # újrakalkuláljuk a koherenciaszintet az új attitude-okkal
                coherence_diff = self.coherency_level - starting_coherency
                if np.exp(coherence_diff / temperature * (i + 1)) > np.random.random():
                    starting_attitudes = self.attitude_list.copy()
                    starting_coherency = self.coherency_level
                else:
                    self.attitude_list = starting_attitudes.copy()
                    self.coherency_level = starting_coherency


spec_2 = [
    ('sn_size', int64),
    ('bs_size', int64),
    ('dissonance_penalty', float64),
    ('const_negative_nodes', int64[:]),
    ('const_positive_nodes', int64[:]),
    ('temperature', float64),
    ('soc_nw', float64[:, :]),
    ('agent_list', types.ListType(Agent.class_type.instance_type)),
    ('use_clones', boolean),
    ('attitude_change_ampl', float64),
    ('soc_incr_ampl', float64),
    ('assoc_incr_ampl', float64),
    ('tca_power', float64),
    ('max_stab', int64),
]


@jitclass(spec_2)
class SocialNetwork:
    """
    The main class describing the social network of agents. Includes many variables governing the communication process.
    """

    def __init__(self, sn_size: int, bs_size: int, const_negative_nodes: List[int], const_positive_nodes: List[int],
                 temperature: float, dissonance_penalty: int,
                 use_clones: bool, attitude_change_ampl: float,
                 soc_incr_ampl: float, assoc_incr_ampl: float, max_stab: int,
                 tca_power: float):
        self.sn_size = sn_size
        self.bs_size = bs_size
        self.const_negative_nodes = const_negative_nodes
        self.const_positive_nodes = const_positive_nodes
        self.temperature = temperature
        self.dissonance_penalty = dissonance_penalty
        self.use_clones = use_clones
        self.attitude_change_ampl = attitude_change_ampl
        self.soc_incr_ampl = soc_incr_ampl
        self.assoc_incr_ampl = assoc_incr_ampl
        self.tca_power = tca_power
        self.max_stab = max_stab
        self.sn_gen()
        self.agent_gen()

    def sn_gen(self):
        """
        Szociális háló előállítása
        """
        soc_nw = np.random.uniform(low=1e-15, high=1.0, size=(self.sn_size, self.sn_size))
        np.fill_diagonal(soc_nw, 0)
        for i in range(self.sn_size):
            for j in range(self.sn_size):
                if soc_nw[i, j] > 0.0:
                    soc_nw[i, j] = 1
                else:
                    soc_nw[i, j] = 0
        self.soc_nw = soc_nw

    def agent_gen(self):
        """
        Ügynökök listájának előállítása
        """
        agent_list = typed.List([Agent(self.bs_size, self.dissonance_penalty, self.const_negative_nodes,
                                       self.const_positive_nodes, self.temperature, self.max_stab,
                                       self.attitude_change_ampl) for _ in
                                 range(self.sn_size)])
        if self.use_clones:
            for index, agent in enumerate(agent_list):
                if index < 1:
                    agent.bs_gen(starter=True)
                else:
                    agent.bs_gen(starter=False, starting_assoc_mtx=agent_list[0].assoc_mtx,
                                 starting_attitudes=agent_list[0].attitude_list)
        else:
            for index, agent in enumerate(agent_list):
                agent.bs_gen(starter=True)
        self.agent_list = agent_list

    def communication(self):
        """
        Main communication process in the social network. This is the main iteration step in the simulation.
        """
        i = np.random.choice(np.arange(self.soc_nw.shape[0]))
        row_i = self.soc_nw[i].copy()
        new_row_i = row_i.copy()
        friends_of_i = [agent_index for agent_index in range(len(row_i)) if row_i[agent_index] > 0]
        for friend_index, friend in enumerate(friends_of_i):
            new_row_i += self.soc_nw[friend] * row_i[friend] / (sum(row_i) ** self.tca_power)
        row_i = new_row_i
        row_i[i] = 0
        if row_i.sum() > 0:
            r = np.random.uniform(low=0.0, high=1.0)
            s = 0
            for item, prob in zip(range(self.soc_nw.shape[1]), row_i / row_i.sum()):
                s += prob
                if s >= r:
                    j = item
                    break
        else:
            r = np.random.uniform(low=0.0, high=1.0, size=self.soc_nw.shape[1])
            r[i] = -1.0
            j = np.argmax(r)
        agent_i = self.agent_list[i]
        agent_j = self.agent_list[j]
        assoc_mtx_adjusted = agent_i.assoc_mtx.copy()

        flat_matrix = assoc_mtx_adjusted.flatten()
        prob = flat_matrix / flat_matrix.sum()
        r = np.random.uniform(low=0.0, high=1.0)
        s = 0
        for item, prob in zip(range(flat_matrix.size), prob):
            s += prob
            if s >= r:
                rand_ind = item
                break
        k = rand_ind % self.bs_size
        rand_ind //= self.bs_size
        l = rand_ind % self.bs_size
        self.agent_list[i].communication_memory = np.roll(self.agent_list[i].communication_memory, 2)
        self.agent_list[i].communication_memory[0] = k
        self.agent_list[i].communication_memory[1] = l
        message = (agent_i.attitude_list[k], agent_i.attitude_list[l])
        opinion = (agent_j.attitude_list[k], agent_j.attitude_list[l])
        agent_j.assoc_mtx[k, l] += random.uniform(0, 1) * self.assoc_incr_ampl
        agent_j.comm_counter[k, l] += 1
        agent_j.comm_counter[l, k] += 1
        if agent_j.assoc_mtx[k, l] < 0:
            agent_j.assoc_mtx[k, l] = 0
        elif agent_j.assoc_mtx[k, l] > 1:
            agent_j.assoc_mtx[k, l] = 1
        agent_j.assoc_mtx[l, k] = agent_j.assoc_mtx[k, l]
        agent_j.assoc_mtx = agent_j.assoc_mtx * agent_j.assoc_sum / agent_j.assoc_mtx.sum()
        agent_j.coherency_level = agent_j.calculate_coherency_level()
        self.agent_list[j] = agent_j

        messaging_change = (message[0] * opinion[0] + message[1] * opinion[1]) * self.soc_incr_ampl
        self.soc_nw[i, j] += random.uniform(0, 1) * messaging_change
        if self.soc_nw[i, j] < 0:
            self.soc_nw[i, j] = 0
        elif self.soc_nw[i, j] > 1:
            self.soc_nw[i, j] = 1
        self.agent_list[j].bs_stab(k, self.temperature)
        self.agent_list[j].bs_stab(l, self.temperature)
        self.agent_list[j].bs_stab(-1, self.temperature)
        opinion_2 = (agent_j.attitude_list[k], agent_j.attitude_list[l])
        messaging_change_2 = (message[0] * opinion_2[0] + message[1] * opinion_2[1]) * self.soc_incr_ampl
        self.soc_nw[j, i] += random.uniform(0, 1) * (messaging_change_2)
        if self.soc_nw[j, i] < 0:
            self.soc_nw[j, i] = 0
        elif self.soc_nw[j, i] > 1:
            self.soc_nw[j, i] = 1


def get_modularity(network: np.array):
    """
    Leiden modularity calculation using iGraph
    """
    graph = ig.Graph()

    num_nodes = network.shape[0]
    graph.add_vertices(num_nodes)
    weights = []
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if network[i, j] != 0:
                weight = network[i, j]
                weights.append(weight)
                edges.append((i, j))
    graph.add_edges(edges)
    graph.es["weight"] = weights
    best_partition = graph.community_leiden(objective_function="modularity", weights=graph.es["weight"])
    largest_component_ratio = graph.clusters().giant().vcount() / graph.vcount()
    component_sizes = [len(component) for component in graph.connected_components()]
    return best_partition.modularity, best_partition, largest_component_ratio, component_sizes


# @njit
def get_group_homogenity(agents: List[Agent], agent_count):
    sums = []
    free_attitude_range = len(agents[0].attitude_list) - len(agents[0].const_positive_nodes) - len(
        agents[0].const_negative_nodes)
    for i in range(agent_count - 1):
        for j in range(i + 1, agent_count):
            sums.append(np.sum(np.absolute(
                agents[i].attitude_list[0:free_attitude_range] - agents[j].attitude_list[0:free_attitude_range])) / (
                                2 * free_attitude_range))
    result = 1 - np.mean(np.array(sums))
    if np.isnan(result):
        raise Exception(f"Nan result in agents")
    return result


def get_average_subgroup_homogenity(agents: List[Agent], membership: List[int]):
    subgroups = {}
    for subgroup in set(membership):
        subgroups[subgroup] = []
    for agent, member in zip(agents, membership):
        subgroups[member].append(agent)
    homogenities = []
    for key, item in subgroups.items():
        if len(item) > 1:
            try:
                homogenities.append(get_group_homogenity(item, len(item)))
            except ZeroDivisionError:
                homogenities.append(np.nan)
    weights = [len(group) for group in subgroups.values() if len(group) > 1]
    try:
        result = np.average(homogenities, weights=weights)
    except ZeroDivisionError:
        result = np.nan
    return result


@njit
def get_bs_weight_homogenity(agents: List[Agent]):
    weight_homogenities = []
    agent_count = len(agents)
    bs_size = agents[0].assoc_mtx.shape[0] * (agents[0].assoc_mtx.shape[0] - 1)
    for i in range(agent_count - 1):
        for j in range(i + 1, agent_count):
            weight_homogenities.append(
                np.sum(np.absolute(agents[i].assoc_mtx.flatten() - agents[j].assoc_mtx.flatten()) / bs_size))
    result = 1 - np.mean(np.array(weight_homogenities))

    return result


def get_average_subgroup_bs_weight_homogenity(agents: List[Agent], membership: List[int]):
    subgroups = {}
    for subgroup in set(membership):
        subgroups[subgroup] = []
    for agent, member in zip(agents, membership):
        subgroups[member].append(agent)
    homogenities = []
    for key, item in subgroups.items():
        if len(item) > 1:
            try:
                homogenities.append(get_bs_weight_homogenity(item))
            except ZeroDivisionError:
                homogenities.append(np.nan)
    weights = [len(group) for group in subgroups.values() if len(group) > 1]
    try:
        result = np.average(homogenities, weights=weights)
    except ZeroDivisionError:
        result = np.nan
    return result


def calculate_extremism(agents):
    extremism = 0
    for agent in agents:
        for attitude in agent.attitude_list:
            extremism += abs(attitude) / len(agent.attitude_list)
    extremism = extremism / len(agents)
    return extremism


def get_average_group_size(membership):
    size = []
    for group in set(membership):
        counter = 0
        for member in membership:
            if member == group:
                counter += 1
        size.append(counter)
    return size


def check_convergence(edge_weights, subgroup_bs_weights, conv_count, social_network,
                      edge_norm, edge_thresh, bs_thresh, min_num_edges):
    if len(edge_weights) < min_num_edges:
        return conv_count
    edge_prev = np.mean(edge_weights[-2000:-1000])
    edge_curr = np.mean(edge_weights[-1000:])
    if abs(edge_curr - edge_prev) / edge_norm < edge_thresh:
        edge_conv = True
    else:
        edge_conv = False
    subgroup_bs_weights_prev = np.mean(subgroup_bs_weights[-2000:-1000])
    subgroup_bs_weights_curr = np.mean(subgroup_bs_weights[-1000:])
    if abs(subgroup_bs_weights_curr - subgroup_bs_weights_prev) / subgroup_bs_weights_prev < bs_thresh:
        bs_weights_conv = True
    else:
        bs_weights_conv = False
    if edge_conv and bs_weights_conv:
        conv_count += 1
    else:
        conv_count = 0
        for agent in social_network.agent_list:
            agent.comm_counter = np.zeros((agent.bs_size, agent.bs_size))
    return conv_count


def run_bsdyn_simulation(N, k, iters, penalties, dirname, tca_power, max_iters=10000000, agent_max_stab=10,
                         soc_incr_ampl=1,
                         attitude_change_ampl=1,
                         assoc_incr_ampl=1, num_const_beliefs=1, temperature=0.01, use_clones=False,
                         convergence_start=200000, convergence_end=200, edge_norm=5000, edge_thresh=0.005,
                         bs_thresh=0.05):
    """
    :param N: Size of the social network
    :param k: Size of the belief system
    :param iters: Number of iterations with each parameter set
    :param penalties: Iterable of dissonance penalty values to be used in run
    :param dirname: Path of directory to save outputs to
    :param tca_power: Exponent of the TCA function
    :param max_iters: Maximum number of iterations to cut off at if there's no convergence detected
    :param agent_max_stab: Number of belief system stabilization steps for agents
    :param soc_incr_ampl: Amplitude of edge weight change in social network
    :param attitude_change_ampl: Amplitude of attitude changes during belief system stabilization
    :param assoc_incr_ampl: Amplitude of association increase as a result of communication
    :param num_const_beliefs: Number of constant belief pairs in the belief system
    :param temperature: Temperature parameter used in the belief system stabilization process
    :param use_clones: Whether to start from clones of identical agents or not
    :param convergence_start: Iteration to start checking for convergence at
    :param convergence_end: Number of convergence steps to stop the simulation
    :param edge_norm: Edge normalization factor in the convergence process
    :param edge_thresh: Value below which we accept fluctuations in edge weights for convergence
    :param bs_thresh: Value below which we accept fluctuatuions in belief system homogeneities for convergence
    """
    for iteration in range(iters):
        for penalty in penalties:
            const_neg_beliefs = [k - num_const_beliefs * 2 + i for i in range(num_const_beliefs)]
            const_pos_beliefs = [k - num_const_beliefs + i for i in range(num_const_beliefs)]
            test_network = SocialNetwork(N, k, np.array(const_neg_beliefs, dtype=np.int64),
                                         np.array(const_pos_beliefs, dtype=np.int64),
                                         temperature, penalty, use_clones, attitude_change_ampl, soc_incr_ampl,
                                         assoc_incr_ampl, agent_max_stab, tca_power,
                                         )

            modularities = []
            num_communities = []
            bs_weight_homogenities = []
            subgroup_bs_weight_homogenities = []
            partitions = []
            num_edges = []
            sum_edge_weights = []
            homogenities = []
            subgroup_homogenities = []
            largest_component_ratios = []
            components = []
            extremism = []
            group_sizes = []
            conv_count = 0
            for i in range(max_iters):
                if i % 100 == 0:
                    index = int((i - 1) / 100)
                    modularity, partition, largest_ratio, component_list = get_modularity(test_network.soc_nw)
                    modularities.append(modularity)
                    largest_component_ratios.append(largest_ratio)
                    components.append(component_list)
                    num_communities.append(len(partition.sizes()))
                    num_edges.append(np.count_nonzero(test_network.soc_nw))
                    sum_edge_weights.append(np.sum(test_network.soc_nw))
                    homogenities.append(get_group_homogenity(test_network.agent_list, N))
                    if i < 1 and test_network.use_clones:
                        assert homogenities[0] == 1.0
                    subgroup_homogenities.append(
                        get_average_subgroup_homogenity(test_network.agent_list, partition.membership))
                    bs_weight_homogenities.append(get_bs_weight_homogenity(test_network.agent_list))
                    subgroup_bs_weight_homogenities.append(
                        get_average_subgroup_bs_weight_homogenity(test_network.agent_list, partition.membership))
                    extremism.append(calculate_extremism(test_network.agent_list))
                    group_sizes.append(get_average_group_size(partition.membership))
                    if i > convergence_start:
                        conv_count = check_convergence(sum_edge_weights, bs_weight_homogenities, conv_count,
                                                       test_network, edge_norm, edge_thresh, bs_thresh,
                                                       int(convergence_start / 100))
                        if conv_count >= convergence_end:
                            break
                    if i % 10000 == 0:
                        partitions.append(partition)
                        print(f'Finished with iteration {i}')
                test_network.communication()

            DIRECTORY_NAME = dirname + f"\soc_inc_{soc_incr_ampl}_dissonance_penalty_{penalty}_bs_size{k}_const_beliefs_{num_const_beliefs}_num_agents_{N}_iter_{iteration}"
            exists = os.path.exists(DIRECTORY_NAME)

            if not exists:
                os.makedirs(DIRECTORY_NAME)

            nonzeros = []
            for elem in np.array([agent.assoc_mtx.flatten() for agent in test_network.agent_list]).flatten():
                if elem != 0:
                    nonzeros.append(elem)

            results_dict = {"Modularities": modularities,
                            "Num_communities": num_communities,
                            "Num_edges": num_edges,
                            "Sum_edges": sum_edge_weights,
                            "Homogenities_total": homogenities,
                            "Subgroup_avg_homogenities": subgroup_homogenities,
                            "Bs_weight_homogenities": bs_weight_homogenities,
                            "Subgroup_avg_bs_weight_homogenities": subgroup_bs_weight_homogenities,
                            "Extremism": extremism,
                            "Group_sizes": group_sizes,
                            "Largest_component_ratios": largest_component_ratios,
                            "Component_list": components,
                            "Final_bss": [agent.assoc_mtx for agent in test_network.agent_list],
                            "Comm_counters": [a.comm_counter for a in test_network.agent_list],
                            "Final_Partition": partition,
                            }

            with open(f"{DIRECTORY_NAME}/results.pkl", "wb") as file:
                pickle.dump(results_dict, file)

            ig.plot(partition, f"{DIRECTORY_NAME}/igraph_plot.pdf", bbox=(600, 600), margin=40)
