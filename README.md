# Assoc-based-bsdyn

This repository contains the main function required for running social network simulations with our association based belief system model.

To run a simulation, call the run_bsdyn_simulation function. This has some required and optional arguments:

Mandatory:
* N: Number of agents in social network - Large social networks can slow down simulation significantly
* k: Size of belief system - Large belief systems can slow down simulation
* iters: Number of runs with each parameter set
* penalties: List of dissonance penalties to use in simulations
* dirname: Directory path to save outputs to
* tca_power: Exponent of the TCA function

Optional:
* max_iters: Maximum number of communication steps without convergence. Default value: 10 000 000
* agent_max_stab: Number of stabilization steps for agents after communication. High values can slow down the simulation. Default value: 10
* soc_incr_ampl: Amplitude of change in social ties after communication. Default value: 1
* attitude_change_ampl: Amplitude of change in attitudes during stabilization process. Default value: 1
* assoc_incr_ampl: Amplitude of change in association strength in the belief system due to communication. Default value: 1
* num_const_beliefs: Number of constant positive and negative beliefs in belief system. Default value: 1
* temperature: Used to govern the probability of accepting non-optimal attitude changes during stabilization process. Default value: 0.01
* use_clones: Boolean variable, governs whether to start with identical agents. Default value: False
* convergence_start: Communication step after which the convergence check starts. Default value: 200 000
* convergence_end: Number of consecutive passed convergence checks required to end simulation. Default value: 200
* edge_norm: Normalization factor for total edge weights during convergence check. Default value: 5000
* edge_thresh: Tolerance range within which we accept the fluctuations in the normalized total edge weights. Default value: 0.005
* bs_thresh: Tolerance range within which we accept the fluctuations in the normalized subgroup belief homogeneities. Default value: 0.05

The exact values used for our simulations can be found in our article. Please consult them as guidelines if you want to try new parameter sets.

## Results

The results produced by the program are placed in the specified directory. It consists of a figure of the final partition of the social network, and a pickled dictionary containing additional information.

The dictionary fields are the following:
* Modularities: List of calculated modularities every 100th iteration step
* Num_communities: List containing the number of communities in the SN every 100th iteration step
* Num_edges: List containing the number of existing edges in the SN every 100th iteraton step
* Sum_edges: List containing the total edge weights in the SN every 100th iteration step
* Homogeneities_total: List containing the attitude homogeneities calculated on the whole social network every 100th iteration step
* Subgroup_avg_homogeneities: List containing the average attitude homogeneities calculated in the communities every 100th iteration step
* Bs_weight_homogenities: List containing the belief system homogeneities calculated on the whole social network every 100th iteration step
* Subgroup_avg_bs_weight_homogenities: List containing the average belief system homogeneities calculated in the communities every 100th iteration step
* Extremism: List containing the average extremism of the agents every 100th iteration step
* Group_sizes: List containing lists of the community sizes every 100th iteration step
* Largest_component_ratios: List containing the ratio of the largest connected component and the total nodes in the social network every 100th iteration step
* Component_list: List of lists containing the sizes of connected components in the social network every 100th iteration step
* Final_bss: List containing the belief system arrays of each agent at the last iteration step
* Comm_counters: List containing the communication counter arrays of each agent at the last iteration step
* Final_partition: igraph VertexClustering object of the last partition detected by the Leiden algorithm
