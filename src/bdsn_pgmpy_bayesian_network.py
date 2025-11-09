import typing
import pandas as pd
import numpy as np
import re
import time
import pgmpy as pgmpy
from pgmpy.models import BayesianNetwork
import pickle
import networkx as nx
from pgmpy.estimators.MLE import MaximumLikelihoodEstimator
from pgmpy.estimators.EM import ExpectationMaximization
from pgmpy.estimators.ExpectationMaximizationWithPriorInformation import PenalizedExpectationMaximization
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pysmile
import pysmile_license

class PgmpyBdsnBayesianNetwork():

    FC_PRIOR_DIRICHLET_DISTRIBUTIONS = {"Wahr": 0.0, "Falsch": 0.0}
    FC_PRIOR_EXPERIENCE_TABLE = {0: 0}
    FC_FIXED_PARAMETERS = {}

    FE_PRIOR_DIRICHLET_DISTRIBUTIONS = {
        "Wahr":
            {
                "Unbekannt": 0.2,
                "OK": 0.05
            },
        "Falsch":
            {
                "Unbekannt": 0.8,
                "OK": 0.95
            }
    }
    FE_PRIOR_EXPERIENCE_TABLE = {"Unbekannt": 5, "OK": 5}
    FE_FIXED_PARAMETERS = {}

    CM_PRIOR_DIRICHLET_DISTRIBUTIONS = {
        "Wahr":
            {
                "Falsch": 0.02
            },
        "Falsch":
            {
                "Falsch": 0.98
            }
    }
    CM_PRIOR_EXPERIENCE_TABLE = {"Falsch": 5}
    CM_FIXED_PARAMETERS = {}

    F_PRIOR_DIRICHLET_DISTRIBUTIONS = {
        "Unbekannt":
            {
                "Wahr": 0.019,
                "Falsch": 1e-7
            },
        "OK":
            {
                "Wahr": 0.001,
                "Falsch": "X"
            }
    }
    F_PRIOR_EXPERIENCE_TABLE = {"Falsch": 10}
    F_FIXED_PARAMETERS = {"Unbekannt": 1, "OK": 1}

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("Instantiated empty Pgmpy BDSN Bayesian Network.")
        self.pgmpy_bayesian_network = pgmpy.models.BayesianNetwork()

    def createBayesianNetworkFromDatasetForEmLearning(self, data: pd.DataFrame):
        col_names = data.columns.to_numpy()
        listFCs = [i for i in col_names if re.search( "FC.._", i) is not None]
        listFEs = [i for i in col_names if re.search( "FE.._", i) is not None]
        listCMs = [i for i in col_names if re.search( "CM.._", i) is not None]
        listFuntions = [i for i in col_names if re.search( "Func.._", i) is not None]

        node_pairs_list = []

    def add_node(self, node, weight=None, latent=False):
        self.pgmpy_bayesian_network.add_node(node=node, weight=weight, latent=latent)

    def add_edge(self, u, v, **kwargs):
        self.pgmpy_bayesian_network.add_edge(u=u, v=v, **kwargs)

    def add_cpds(self, *kwargs):
        self.pgmpy_bayesian_network.add_cpds(*kwargs)

    def defineCPD(self,
                  variable,
                  variable_card,
                  values,
                  evidence=None,
                  evidence_card=None,
                  state_names={}):
        return pgmpy.factors.discrete.TabularCPD(variable=variable,
                                                 variable_card=variable_card,
                                                 values=values,
                                                 evidence=evidence,
                                                 evidence_card=evidence_card,
                                                 state_names=state_names)

    def printDAG(self):
        G = self.pgmpy_bayesian_network.copy()
        for layer, nodes in enumerate(nx.topological_generations(G)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                G.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(G,
                                     subset_key="layer",
                                     align="horizontal")

        fig, ax = plt.subplots()
        nx.draw_networkx(G,
                         pos=pos,
                         ax=ax,
                         arrowsize=30,
                         node_size=500
                         )
        ax.set_title("DAG layout in topological order")
        fig.tight_layout()
        plt.show()

        '''        
        nx.draw_shell(
            self.pgmpy_bayesian_network,
            with_labels=True,
            arrowsize=30,
            node_size=500,
            alpha=0.6,
            font_weight="bold"
        )
        plt.show()
        '''

        return 0

    def parameterLearningMLE(self, data_samples) -> typing.List:
        mle = MaximumLikelihoodEstimator(model=self.pgmpy_bayesian_network,
                                         data=data_samples)

        # Estimating the CPD for a single node.
        #print(mle.estimate_cpd(node="FIO2"))
        #print(mle.estimate_cpd(node="CVP"))

        # Estimating CPDs for all the nodes in the model
        return mle.get_parameters()


    def parameterLearningEM(self, data_samples, latent_card=None):
        em = ExpectationMaximization(model=self.pgmpy_bayesian_network,
                                     data=data_samples,
                                     state_names=self.pgmpy_bayesian_network.states)
        em_learned_parameters= em.get_parameters(latent_card=latent_card)
        return em_learned_parameters

    def parameterLearningPenalizedEM(self,
                                     data_samples,
                                     latent_card=None,
                                     #state_names=None,
                                     initialization_method="random",
                                     inference_algorithm="variable_elimination",
                                     dirichlet_probability_distributions=None,
                                     experience_table=None,
                                     fixed_parameters=None,
                                     delta=0.01,
                                     max_iter=50):
        pem = PenalizedExpectationMaximization(
            model=self.pgmpy_bayesian_network,
            data=data_samples,
            dirichlet_probability_distributions=dirichlet_probability_distributions,
            experience_table=experience_table,
            fixed_parameters=fixed_parameters,
            max_iter=max_iter,
            delta=delta,
            initalization_method=initialization_method,
            state_names=self.pgmpy_bayesian_network.states,
            inference_algorithm=inference_algorithm
        )
        dict_cpds_t_p_1, list_calculated_log_likelihood_scores, list_calculated_historic_cpds = pem.get_parameters(latent_card=latent_card)
        return dict_cpds_t_p_1, list_calculated_log_likelihood_scores, list_calculated_historic_cpds

    def getPriorInformations(self):
        """

        :return: dirichlet_probability_distributions, experience_table, fixed_parameters
        """
        initialized_probability_parameters = {}
        dirichlet_probability_distributions = {}
        experience_table = {}
        fixed_parameters = {}

        for cpd in self.pgmpy_bayesian_network.cpds:
            var_name = cpd.variable
            cardinality = cpd.cardinality
            state_names = cpd.state_names[var_name]

            if "FC" in var_name:
                # initialize the probability parameters with uniform distributions
                init_params = np.zeros_like(cpd.values)
                uniform_val = 1.0 / init_params.shape[0]
                for ci in range(init_params.shape[0]):
                    init_params[ci] = uniform_val

                # define the prior Dirichlet distributions
                cur_dir_dist = np.zeros_like(cpd.values)
                for si in state_names:
                    cur_val = PgmpyBdsnBayesianNetwork.FC_PRIOR_DIRICHLET_DISTRIBUTIONS[si]
                    cpos = cpd.name_to_no[var_name][si]
                    cur_dir_dist[cpos] = cur_val

                # define the experience table
                cur_exp_tab = np.zeros(shape=cpd.values.shape[1:] if len(cpd.values.shape) > 1 else 1)
                for ci in range(cur_exp_tab.shape[0]):
                    cur_exp_tab[ci] = PgmpyBdsnBayesianNetwork.FC_PRIOR_EXPERIENCE_TABLE[0]

                # define the fixed parameters
                cur_fixed_params = np.zeros_like(cpd.values)

            elif "FE" in var_name:
                # initialize the probability parameters with uniform distributions
                init_params = np.full(shape=cpd.values.shape, fill_value=0.5)
                # define the prior Dirichlet distributions
                cur_dir_dist = np.zeros_like(cpd.values)
                # define the fixed parameters
                cur_fixed_params = np.zeros_like(cpd.values)
                # define the experience table
                cur_exp_tab = np.zeros(shape=cpd.values.shape[1:] if len(cpd.values.shape) > 1 else 1)

                for idx in np.ndindex(cpd.values.shape):
                    pa_states = []
                    for inum, ipos in enumerate(idx[1:]):
                        pa_states.append(cpd.no_to_name[list(cpd.state_names.keys())[inum + 1]][ipos])

                    if state_names[idx[0]] == "Wahr" and "OK" in pa_states:
                        cur_dir_dist[idx] = 0.05
                    elif state_names[idx[0]] == "Wahr" and "Unbekannt" in pa_states:
                        cur_dir_dist[idx] = 0.2
                    elif state_names[idx[0]] == "Falsch" and "OK" in pa_states:
                        cur_dir_dist[idx] = 0.95
                    elif state_names[idx[0]] == "Falsch" and "Unbekannt" in pa_states:
                        cur_dir_dist[idx] = 0.8

                    if len(np.unique(np.asarray(pa_states))) == 1 and "OK" in pa_states:
                        cur_exp_tab[idx[1:]] = 100 #10
                    elif len(np.unique(np.asarray(pa_states))) == 1 and "Unbekannt" in pa_states:
                        cur_exp_tab[idx[1:]] = 100 #10


            elif "CM" in var_name:
                # initialize the probability parameters with uniform distributions
                init_params = np.zeros_like(cpd.values)
                # define the prior Dirichlet distributions
                cur_dir_dist = np.zeros_like(cpd.values)
                # define the fixed parameters
                cur_fixed_params = np.zeros_like(cpd.values)
                # define the experience table
                cur_exp_tab = np.zeros(shape=cpd.values.shape[1:] if len(cpd.values.shape) > 1 else 1)

                uniform_val = 1. / init_params.shape[0]
                for idx in np.ndindex(cpd.values.shape):
                    pa_states = []
                    for inum, ipos in enumerate(idx[1:]):
                        pa_states.append(cpd.no_to_name[list(cpd.state_names.keys())[inum + 1]][ipos])

                    if state_names[idx[0]] == "Wahr" and "Wahr" in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 1.0
                    elif state_names[idx[0]] == "Falsch" and "Wahr" in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 0.0
                    elif state_names[idx[0]] == "Wahr" and "Wahr" not in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 0.02
                    elif state_names[idx[0]] == "Falsch" and "Wahr" not in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 0.98

                    if "Wahr" not in pa_states:
                        cur_exp_tab[idx[1:]] = 100 #10
                    elif "Wahr" in pa_states:
                        cur_exp_tab[idx[1:]] = 50 #4

            elif "Func" in var_name or (("SE" in var_name and "Komponente" in var_name) or ("SE" in var_name and "Prozess" in var_name)):
                # initialize the probability parameters with uniform distributions
                init_params = np.zeros_like(cpd.values)
                # define the prior Dirichlet distributions
                cur_dir_dist = np.zeros_like(cpd.values)
                # define the fixed parameters
                cur_fixed_params = np.zeros_like(cpd.values)
                # define the experience table
                cur_exp_tab = np.zeros(shape=cpd.values.shape[1:] if len(cpd.values.shape) > 1 else 1)

                uniform_val = (1.0 - 0.001 - 0.019)  / (init_params.shape[0] - 2)
                for idx in np.ndindex(cpd.values.shape):
                    pa_states = []
                    for inum, ipos in enumerate(idx[1:]):
                        pa_states.append(cpd.no_to_name[list(cpd.state_names.keys())[inum+1]][ipos])

                    if state_names[idx[0]] == "OK" and "Wahr" in pa_states:
                        init_params[idx] = 0.001
                        cur_fixed_params[idx] = 1
                    elif state_names[idx[0]] == "Unbekannt" and "Wahr" in pa_states:
                        init_params[idx] = 0.019
                        cur_fixed_params[idx] = 1
                    elif state_names[idx[0]] == "OK" and "Wahr" not in pa_states:
                        init_params[idx] = 1. / init_params.shape[0]
                        cur_dir_dist[idx] = 1 - (cur_dir_dist.shape[0]-1)*1e-7
                    elif state_names[idx[0]] == "Unbekannt" and "Wahr" not in pa_states:
                        init_params[idx] = 1. / init_params.shape[0]
                        cur_dir_dist[idx] = 1e-7
                    else:
                        if "Wahr" in pa_states:
                            init_params[idx] = uniform_val
                        else:
                            init_params[idx] = 1. / init_params.shape[0]
                            cur_dir_dist[idx] = 1e-7

                    if "Wahr" not in pa_states:
                        cur_exp_tab[idx[1:]] = 100 # 10
            else:
                raise ValueError("Houston, we should not be here!")

            initialized_probability_parameters[var_name] = init_params
            dirichlet_probability_distributions[var_name] = cur_dir_dist
            experience_table[var_name] = cur_exp_tab
            fixed_parameters[var_name] = cur_fixed_params

        return initialized_probability_parameters, dirichlet_probability_distributions, experience_table, fixed_parameters

    def getUniformProbabilityParameters(self):
        """
        :return: uniformly distributed parameters
        """
        initialized_probability_parameters = {}
        for cpd in self.pgmpy_bayesian_network.cpds:
            var_name = cpd.variable

            if "FC" in var_name:
                # initialize the probability parameters with uniform distributions
                init_params = np.zeros_like(cpd.values)
                uniform_val = 1.0 / init_params.shape[0]
                for ci in range(init_params.shape[0]):
                    init_params[ci] = uniform_val

            elif "FE" in var_name:
                # initialize the probability parameters with uniform distributions
                init_params = np.full(shape=cpd.values.shape, fill_value=0.5)

            elif "CM" in var_name:
                # initialize the probability parameters with uniform distributions
                init_params = np.full(shape=cpd.values.shape, fill_value=0.5)
                """
                init_params = np.zeros_like(cpd.values)
                # define the prior Dirichlet distributions
                cur_dir_dist = np.zeros_like(cpd.values)
                # define the fixed parameters
                cur_fixed_params = np.zeros_like(cpd.values)
                # define the experience table
                cur_exp_tab = np.zeros(shape=cpd.values.shape[1:] if len(cpd.values.shape) > 1 else 1)

                uniform_val = 1. / init_params.shape[0]
                for idx in np.ndindex(cpd.values.shape):
                    pa_states = []
                    for inum, ipos in enumerate(idx[1:]):
                        pa_states.append(cpd.no_to_name[list(cpd.state_names.keys())[inum + 1]][ipos])

                    if state_names[idx[0]] == "Wahr" and "Wahr" in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 1.0
                    elif state_names[idx[0]] == "Falsch" and "Wahr" in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 0.0
                    elif state_names[idx[0]] == "Wahr" and "Wahr" not in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 0.02
                    elif state_names[idx[0]] == "Falsch" and "Wahr" not in pa_states:
                        init_params[idx] = uniform_val
                        cur_dir_dist[idx] = 0.98

                    if "Wahr" not in pa_states:
                        cur_exp_tab[idx[1:]] = 100 #10
                    elif "Wahr" in pa_states:
                        cur_exp_tab[idx[1:]] = 50 #4
                """

            elif "Func" in var_name or (("SE" in var_name and "Komponente" in var_name) or ("SE" in var_name and "Prozess" in var_name)):
                # initialize the probability parameters with uniform distributions
                init_params = np.zeros_like(cpd.values)
                uniform_val = (1.0) / (init_params.shape[0])
                init_params = np.full(shape=cpd.values.shape, fill_value=uniform_val)
                """
                for idx in np.ndindex(cpd.values.shape):
                    pa_states = []
                    for inum, ipos in enumerate(idx[1:]):
                        pa_states.append(cpd.no_to_name[list(cpd.state_names.keys())[inum+1]][ipos])

                    if state_names[idx[0]] == "OK" and "Wahr" in pa_states:
                        init_params[idx] = 0.001
                    elif state_names[idx[0]] == "Unbekannt" and "Wahr" in pa_states:
                        init_params[idx] = 0.019
                        cur_fixed_params[idx] = 1
                    elif state_names[idx[0]] == "OK" and "Wahr" not in pa_states:
                        init_params[idx] = 1. / init_params.shape[0]
                        cur_dir_dist[idx] = 1 - (cur_dir_dist.shape[0]-1)*1e-7
                    elif state_names[idx[0]] == "Unbekannt" and "Wahr" not in pa_states:
                        init_params[idx] = 1. / init_params.shape[0]
                        cur_dir_dist[idx] = 1e-7
                    else:
                        if "Wahr" in pa_states:
                            init_params[idx] = uniform_val
                        else:
                            init_params[idx] = 1. / init_params.shape[0]
                            cur_dir_dist[idx] = 1e-7

                    if "Wahr" not in pa_states:
                        cur_exp_tab[idx[1:]] = 100 # 10
                """
            else:
                raise ValueError("Houston, we should not be here!")

            initialized_probability_parameters[var_name] = init_params

        return initialized_probability_parameters

    def pgmpyCpdDefinitionToSmileCpdDefinition(self, node):
        # Todo implement
        return 0

    def getBayesianNetwork(self):
        return self.pgmpy_bayesian_network

    def setBayesianNetwork(self, bn):
        self.pgmpy_bayesian_network = bn

    def createCptNode(self,
                      id: str,
                      name: str,
                      bdsn_node_type: int = None,
                      node_states: typing.List = [],
                      x_pos: int = None,
                      y_pos: int = None,
                      width: int = 85,
                      height: int = 85):
        handle = self.bayesian_network.add_node(node_type=pysmile.NodeType.NOISY_MAX if bdsn_node_type == BdsnBayesianNetwork.FUNCTIONAL_UNIT_NODE else pysmile.NodeType.CPT,
                                                node_id=id)
        self.bayesian_network.set_node_name(handle, name)
        self.bayesian_network.set_node_position(handle, x_pos, y_pos, width, height)
        initial_outcome_count = self.bayesian_network.get_outcome_count(handle)
        self.bayesian_network.set_node_bg_color(handle, BdsnBayesianNetwork.MAP_NODE_TYP_COLOR[bdsn_node_type])

        for i in range(0, initial_outcome_count):
            self.bayesian_network.set_outcome_id(handle, i, node_states[i])

        for i in range(initial_outcome_count, len(node_states)):
            self.bayesian_network.add_outcome(handle, node_states[i])

        return handle


    def createDiscreteDeterministicNode(self,
                                        id: str,
                                        name: str,
                                        bdsn_node_type: int = None,
                                        node_states: typing.List = [],
                                        x_pos: int = None,
                                        y_pos: int = None,
                                        width: int = 85,
                                        height: int = 85):
        handle = self.bayesian_network.add_node(node_type=pysmile.NodeType.TRUTH_TABLE, node_id=id)
        self.bayesian_network.set_node_name(handle, name)
        self.bayesian_network.set_node_position(handle, x_pos, y_pos, width, height)
        initial_outcome_count = self.bayesian_network.get_outcome_count(handle)
        self.bayesian_network.set_node_bg_color(handle, BdsnBayesianNetwork.MAP_NODE_TYP_COLOR[bdsn_node_type])

        for i in range(0, initial_outcome_count):
            self.bayesian_network.set_outcome_id(handle, i, node_states[i])

        for i in range(initial_outcome_count, len(node_states)):
            self.bayesian_network.add_outcome(handle, node_states[i])

        return handle


    def saveBayesianNetworkToFile(self, path: str):
        self.bayesian_network.write_file(path)
        self.logger.info(f"Have written Bayesian network successfully to {path} XDSL file.")


    def addArc(self, handle_node_parent, handle_node_child):
        self.bayesian_network.add_arc(handle_node_parent, handle_node_child)


    def getNodeHandle(self, id):
        return self.bayesian_network.get_node(id)


    def initializeDiscreteStructuralElementNodes(self, handle_node):
        list_parent_node_ids = self.bayesian_network.get_parent_ids(handle_node)
        list_parent_node_handles = self.bayesian_network.get_parents(handle_node)

        if len(list_parent_node_ids) == 0:
            return

        cpt = self.bayesian_network.get_node_definition(handle_node)
        dim_count = 1 + len(list_parent_node_handles)
        dim_sizes = [0] * dim_count

        for i in range(0, dim_count - 1):
            dim_sizes[i] = self.bayesian_network.get_outcome_count(list_parent_node_handles[i])

        dim_sizes[len(dim_sizes) - 1] = self.bayesian_network.get_outcome_count(handle_node)
        coords = [0] * dim_count

        for elem_idx in range(0, len(cpt)):
            self.indexToCoords(elem_idx, dim_sizes, coords)
            outcome = self.bayesian_network.get_outcome_id(handle_node, coords[dim_count - 1])
            if dim_count > 1:
                child_outcomes = []
                for parent_idx in range(0, len(list_parent_node_handles)):
                    parent_handle = list_parent_node_handles[parent_idx]
                    child_outcomes.append(self.bayesian_network.get_outcome_id(parent_handle, coords[parent_idx]))

            if  "NOK" not in child_outcomes and outcome == "OK":
                cpt[elem_idx] = 1.0
            elif "NOK" not in child_outcomes and outcome == "NOK":
                cpt[elem_idx] = 0.0
            elif "NOK" in child_outcomes and outcome == "NOK":
                cpt[elem_idx] = 1.0
            elif "NOK" in child_outcomes and outcome == "OK":
                cpt[elem_idx] = 0.0
            else:
                print("Why are we here?")

        print("#################")
        self.bayesian_network.set_node_definition(handle_node, cpt)
        self.printNodeInfo(self.bayesian_network.get_node(handle_node))
        print(cpt)
        print("=================")


    def printNodeInfo(self, node_handle):
        print("Node id/name: " + self.bayesian_network.get_node_id(node_handle) + "/" + self.bayesian_network.get_node_name(node_handle))
        print("  Outcomes: " + " ".join(self.bayesian_network.get_outcome_ids(node_handle)))
        parent_ids = self.bayesian_network.get_parent_ids(node_handle)

        if len(parent_ids) > 0:
            print("  Parents: " + " ".join(parent_ids))

        child_ids = self.bayesian_network.get_child_ids(node_handle)

        if len(child_ids) > 0:
            print("  Children: " + " ".join(child_ids))

        self.printCptMatrix(node_handle)

    def printCptMatrix(self, node_handle):
        cpt = self.bayesian_network.get_node_definition(node_handle)
        parents = self.bayesian_network.get_parents(node_handle)
        dim_count = 1 + len(parents)
        dim_sizes = [0] * dim_count

        for i in range(0, dim_count - 1):
            dim_sizes[i] = self.bayesian_network.get_outcome_count(parents[i])

        dim_sizes[len(dim_sizes) - 1] = self.bayesian_network.get_outcome_count(node_handle)
        coords = [0] * dim_count

        for elem_idx in range(0, len(cpt)):
            self.indexToCoords(elem_idx, dim_sizes, coords)
            outcome = self.bayesian_network.get_outcome_id(node_handle, coords[dim_count - 1])
            out_str = "    P(" + outcome
            if dim_count > 1:
                out_str += " | "
                for parent_idx in range(0, len(parents)):
                    if parent_idx > 0:
                        out_str += ","

                    parent_handle = parents[parent_idx]
                    out_str += self.bayesian_network.get_node_id(parent_handle) + "=" + self.bayesian_network.get_outcome_id(parent_handle, coords[parent_idx])

            prob = cpt[elem_idx]
            out_str += ")=" + str(prob)
            print(out_str)

    def printCPDs(self):
        for cpd in self.pgmpy_bayesian_network.cpds:
            #if "Komponente_Fettstation" in cpd.variable:
            #    print()
            #if "Func02_Fett_absaugen" in cpd.variable:
            #    print()
            print(f"CPD of variable {str(cpd.variable)}:")
            print(cpd)
            print("")

    @staticmethod
    def plotHistoricLogLikelihoodScores(scores):
        plt.figure()
        x = np.arange(len(scores))
        y = np.asarray(scores)
        plt.plot(x, y)
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Likelihood Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plotHistoricEvaluationData(list_log_likelihood_scores, list_historic_cpds, **kwargs):
        num_of_cpds = len(list_historic_cpds[0])
        fig, axs = plt.subplots(num_of_cpds+1, 1, sharex=True)
        x = np.arange(1, len(list_log_likelihood_scores)+1)
        x_cpds = np.arange(0, len(list_historic_cpds))
        y = np.asarray(list_log_likelihood_scores)

        novel_list = None
        dict_mapping_names = None
        if "shorten_names" in kwargs:
            if kwargs["shorten_names"]:
                novel_list = []
                dict_mapping_names = {}
                # kürzen der Namen
                for le in list_historic_cpds:
                    novel_dict = dict()
                    for k, de in le.items():
                        cur_name = de.variable
                        cur_name = cur_name[0:cur_name.find("_")]
                        cur_cpd = de.copy()
                        cur_cpd.variable = cur_name
                        novel_dict[cur_name] = cur_cpd
                        dict_mapping_names[de.variable] = cur_name

                        states = de.state_names[k]
                        for state in states:
                            if "_" in state:
                                dict_mapping_names[state] = state[0:state.find("_")]
                            else:
                                dict_mapping_names[state] = state

                    novel_list.append(novel_dict)

            #list_historic_cpds = novel_list

        axs[0].plot(x, y)
        axs[0].set_ylabel('Log Likelihood', fontsize=12)
        #axs[0].set_xlabel('EM Iteration')
        axs[0].grid(True)

        cpd_params_data = {}
        for k in list_historic_cpds[0].keys():
            cpd_params_data[k] = {}
            for sk in list_historic_cpds[0][k].state_names[k]:
                var_name = f"theta_{k if dict_mapping_names is None else dict_mapping_names[k]}={sk if dict_mapping_names is None else dict_mapping_names[sk]}"
                if list_historic_cpds[0][k].is_root_node():
                    cpd_params_data[k][var_name] = []
                else:
                    var_name += "|"
                    pa_nodes = set(list_historic_cpds[0][k].state_names).difference(set([k]))
                    card_ = [list_historic_cpds[0][pan].cardinality[0] for pan in pa_nodes]
                    for idx in np.ndindex(tuple(card_)):
                        cur_var_name = var_name
                        for ix, jdx in enumerate(idx):
                            cur_pa = list(pa_nodes)[ix]
                            cur_pa_state = jdx
                            cur_pa_state_name = list_historic_cpds[0][cur_pa].no_to_name[cur_pa][cur_pa_state]
                            cur_var_name += f"{cur_pa if dict_mapping_names is None else dict_mapping_names[cur_pa]}={cur_pa_state_name if dict_mapping_names is None else dict_mapping_names[cur_pa_state_name]}"
                            if ix < len(idx) - 1:
                                cur_var_name += ","

                        cpd_params_data[k][cur_var_name] = []


        for hcpd in list_historic_cpds:
            for k in hcpd.keys():
                for sk in hcpd[k].state_names[k]:
                    #var_name = f"theta_{k}={sk}"
                    var_name = f"theta_{k if dict_mapping_names is None else dict_mapping_names[k]}={sk if dict_mapping_names is None else dict_mapping_names[sk]}"
                    if hcpd[k].is_root_node():
                        cpd_params_data[k][var_name].append(hcpd[k].values[hcpd[k].name_to_no[k][sk]])
                    else:
                        var_name += "|"
                        pa_nodes = set(hcpd[k].state_names).difference(set([k]))
                        card_ = [hcpd[pan].cardinality[0] for pan in pa_nodes]
                        for idx in np.ndindex(tuple(card_)):
                            cur_var_name = var_name
                            for ix, jdx in enumerate(idx):
                                cur_pa = list(pa_nodes)[ix]
                                cur_pa_state = jdx
                                cur_pa_state_name = list_historic_cpds[0][cur_pa].no_to_name[cur_pa][cur_pa_state]
                                #cur_var_name += f"{cur_pa}={cur_pa_state_name}"
                                cur_var_name += f"{cur_pa if dict_mapping_names is None else dict_mapping_names[cur_pa]}={cur_pa_state_name if dict_mapping_names is None else dict_mapping_names[cur_pa_state_name]}"
                                if ix < len(idx) - 1:
                                    cur_var_name += ","

                            cidx = (hcpd[k].name_to_no[k][sk],) + idx
                            cpd_params_data[k][cur_var_name].append(hcpd[k].values[cidx])

        for ic in range(num_of_cpds):
            cur_key = list(cpd_params_data.keys())[ic]
            for param in cpd_params_data[cur_key].keys():
                axs[1+ic].plot(x_cpds, cpd_params_data[cur_key][param], label=param)

            axs[1+ic].set_ylabel(f"CPD Werte von {cur_key}", fontsize=12)
            axs[1+ic].legend()
            axs[1+ic].grid(True)

        axs[1+ic].set_xlabel("Iterationsschritt Penalized EM Algorithmus mit fixierten Parametern", fontsize=12)

        plt.tight_layout()
        plt.show()

    def performRepeatedStratifiedKFoldCrossValidation(self,
                                                      df_data,
                                                      num_cv_iterations,
                                                      n_fold,
                                                      initialized_probability_parameters,
                                                      dirichlet_probability_distributions,
                                                      experience_table,
                                                      fixed_parameters,
                                                      uniform_probability_parameters):
        # n x stratified k-fold cross validation
        from sklearn.model_selection import KFold
        from sklearn.model_selection import StratifiedKFold

        num_cv_iterations = num_cv_iterations
        n_fold = n_fold
        df_all_dataset_em_learning = df_data.copy()
        #y = df_all_dataset_em_learning["Func01_Fett_absaugen"].copy()  # for the stratification of the 10-fold cv based on a function # TODO maybe adapt to functional units
        ###
        rel_cols_for_fold_split = [colx for colx in df_all_dataset_em_learning.columns if "Func" in colx or "Komponente" in colx]
        rel_df_for_fold_split = df_all_dataset_em_learning[rel_cols_for_fold_split].copy()
        list_split_labels = []
        for idx_fold_split, row_fold_split in rel_df_for_fold_split.iterrows():
            qrfs = row_fold_split.index[(row_fold_split != "OK") & (row_fold_split != "Unbekannt")].to_list()
            #qrfs = row_fold_split.index[(row_fold_split != "OK") & (row_fold_split != "Unbekannt") & (row_fold_split != "*")].to_list()
            if len(qrfs) > 1:
                raise ValueError("Big Problems encountered!")
            list_split_labels.append(qrfs[0])
        ###
        y = np.asarray(list_split_labels)
        dict_cv_results = dict()

        for cv_iteration in range(num_cv_iterations):
            print(f"CV Iteration No. {cv_iteration}:")
            #stratified_n_fold_cross_validator = KFold(n_splits=n_fold, random_state=None, shuffle=True)
            stratified_n_fold_cross_validator = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=True)
            #stratified_n_fold_cross_validator.get_n_splits(df_all_dataset_em_learning)
            n_splits = stratified_n_fold_cross_validator.get_n_splits(X=df_all_dataset_em_learning, y=y, groups=None)
            print("\t" + str(stratified_n_fold_cross_validator))
            dict_cv_results[f"cv_iteration_{cv_iteration}"] = {}

            for i_fold, (train_index, test_index) in enumerate(stratified_n_fold_cross_validator.split(X=df_all_dataset_em_learning, y=y)):
                try:
                    print(f"\tFold {i_fold}:")
                    print(f"\t\tTrain: index={train_index}")
                    print(f"\t\tTest:  index={test_index}")
                    df_test_data = df_all_dataset_em_learning.iloc[test_index].copy()
                    df_train_data = df_all_dataset_em_learning.iloc[train_index].copy()

                    dict_query_sets, dict_evidence_sets, dict_query_y = self.extractQueryAndEvidenceSets(df_data=df_test_data)

                    # evalute accuracy with df_test_data
                    eval_metrics = {
                        "accuracy": self.predictionAccuracy,
                        "brier_score": self.predictionBrierScore
                    }

                    dict_cv_results[f"cv_iteration_{cv_iteration}"][f"i_fold_{i_fold}"] = {}
                    cur_place = dict_cv_results[f"cv_iteration_{cv_iteration}"][f"i_fold_{i_fold}"]
                    cur_place["em"] = {}
                    #cur_place["pem"] = {}
                    cur_place["pemfp"] = {}

                    """ 2024-12-02 commented out for test
                    # ==================================================================================================== #
                    # Standard EM                                                                                          #
                    # ==================================================================================================== #
                    # initialize BN with the initialized probability parameters (uniform distribution for EM)
                    params_for_initialization_em = uniform_probability_parameters.copy()
                    for cpd in self.pgmpy_bayesian_network.cpds:
                        cpd.set_values(params_for_initialization_em[cpd.variable])
                    self.printCPDs()

                    t_start_em = time.time_ns()
                    t_start_em_2 = time.perf_counter_ns()
                    dict_em_cpds_t_p_1, list_em_calculated_log_likelihood_scores, list_em_calculated_historic_cpds \
                        = self.parameterLearningPenalizedEM(data_samples=df_train_data,
                                                            latent_card={"B": 2, "C": 2},
                                                            initialization_method="fixed",
                                                            inference_algorithm="variable_elimination",
                                                            delta=0.0001,
                                                            max_iter=50,
                                                            dirichlet_probability_distributions=None,
                                                            experience_table=None,
                                                            fixed_parameters=None
                                                            )
                    t_end_em = time.time_ns()
                    t_end_em_2 = time.perf_counter_ns()
                    delta_t_em = (t_end_em - t_start_em)*1e-9
                    delta_t_em_2 = (t_end_em_2 - t_start_em_2)*1e-9

                    em_bn = self.pgmpy_bayesian_network.copy()
                    # set parameters of the em_bn to the new parameters learned via EM
                    for cpd in em_bn.cpds:
                        cpd.set_values(dict_em_cpds_t_p_1[cpd.variable].values)

                    # save the trained BN
                    em_bn.save(f'./results_for_dissertation_find_the_fault_try_2024-12-02/trained_bns/em_bn_cvitr{cv_iteration}_ifold{i_fold}.net', filetype='net')

                    dict_prediction_metrics_em = self.calculatePredictionMetrics(bn=em_bn,
                                                                                 dict_query_sets=dict_query_sets,
                                                                                 dict_evidence_sets=dict_evidence_sets,
                                                                                 dict_query_y=dict_query_y,
                                                                                 eval_metrics=eval_metrics,
                                                                                 backend="smile")

                    cur_place["em"]["prediction_metrics_of_iteration"] = dict_prediction_metrics_em
                    cur_place["em"]["list_em_calculated_log_likelihood_scores"] = list_em_calculated_log_likelihood_scores
                    cur_place["em"]["list_em_calculated_historic_cpds"] = list_em_calculated_historic_cpds
                    cur_place["em"]["execution_time_time"] = delta_t_em
                    cur_place["em"]["execution_time_perf_counter"] = delta_t_em_2
                    """

                    # ==================================================================================================== #
                    # Penalized EM without fixed parameters                                                                #
                    # ==================================================================================================== #
                    """
                    t_start_pem = time.time_ns()
                    t_start_pem_2 = time.perf_counter_ns()
                    dict_pem_cpds_t_p_1, list_pem_calculated_log_likelihood_scores, list_pem_calculated_historic_cpds \
                        = self.parameterLearningPenalizedEM(data_samples=df_train_data,
                                                            latent_card={"B": 2, "C": 2},
                                                            initialization_method="fixed",
                                                            inference_algorithm="variable_elimination",
                                                            delta=0.0001,
                                                            max_iter=50,
                                                            dirichlet_probability_distributions=dirichlet_probability_distributions,
                                                            experience_table=experience_table,
                                                            fixed_parameters=None
                                                            )
                    t_end_pem = time.time_ns()
                    t_end_pem_2 = time.perf_counter_ns()
                    delta_t_pem = (t_end_pem - t_start_pem)*1e-9
                    delta_t_pem_2 = (t_end_pem_2 - t_start_pem_2)*1e-9

                    pem_bn = self.pgmpy_bayesian_network.copy()
                    # set parameters of the em_bn to the new parameters learned via EM
                    for cpd in pem_bn.cpds:
                        cpd.set_values(dict_pem_cpds_t_p_1[cpd.variable].values)
                    
                    # save the trained BN
                    pem_bn.save(f'./results_for_dissertation/trained_bns/pem_bn_cvitr{cv_iteration}_ifold{i_fold}.net', filetype='net')
                    
                    dict_prediction_metrics_pem = self.calculatePredictionMetrics(bn=pem_bn,
                                                                                 dict_query_sets=dict_query_sets,
                                                                                 dict_evidence_sets=dict_evidence_sets,
                                                                                 dict_query_y=dict_query_y,
                                                                                 eval_metrics=eval_metrics,
                                                                                  backend="smile")

                    cur_place["pem"]["prediction_metrics_of_iteration"] = dict_prediction_metrics_pem
                    cur_place["pem"]["list_pem_calculated_log_likelihood_scores"] = list_pem_calculated_log_likelihood_scores
                    cur_place["pem"]["list_pem_calculated_historic_cpds"] = list_pem_calculated_historic_cpds
                    cur_place["pem"]["execution_time_time"] = delta_t_pem
                    cur_place["pem"]["execution_time_perf_counter"] = delta_t_pem_2
                    """
                    # ==================================================================================================== #
                    # Penalized EM with fixed parameters                                                                   #
                    # ==================================================================================================== #
                    # initialize BN with the initialized probability parameters (uniform distribution with adapted parameters
                    # for PEMFP)
                    params_for_initialization_pemfp = initialized_probability_parameters.copy()
                    for cpd in self.pgmpy_bayesian_network.cpds:
                        cpd.set_values(params_for_initialization_pemfp[cpd.variable])
                    self.printCPDs()

                    t_start_pemfp = time.time_ns()
                    t_start_pemfp_2 = time.perf_counter_ns()
                    dict_pem_w_fp_cpds_t_p_1, list_pem_w_fp_calculated_log_likelihood_scores, list_pem_w_fp_calculated_historic_cpds \
                        = self.parameterLearningPenalizedEM(data_samples=df_train_data,
                                                            latent_card=None, #{"B": 2, "C": 2},
                                                            initialization_method="fixed",
                                                            inference_algorithm="variable_elimination",
                                                            delta=0.0001,
                                                            max_iter=50,
                                                            dirichlet_probability_distributions=dirichlet_probability_distributions,
                                                            experience_table=experience_table,
                                                            fixed_parameters=fixed_parameters
                                                            )
                    t_end_pemfp = time.time_ns()
                    t_end_pemfp_2 = time.perf_counter_ns()
                    delta_t_pemfp = (t_end_pemfp - t_start_pemfp)*1e-9
                    delta_t_pemfp_2 = (t_end_pemfp_2 - t_start_pemfp_2)*1e-9

                    pemfp_bn = self.pgmpy_bayesian_network.copy()
                    # set parameters of the em_bn to the new parameters learned via EM
                    for cpd in pemfp_bn.cpds:
                        cpd.set_values(dict_pem_w_fp_cpds_t_p_1[cpd.variable].values)

                    # save the trained BN
                    pemfp_bn.save(f'./results_for_dissertation_find_the_fault_try_2024-12-02/trained_bns/pemfp_bn_cvitr{cv_iteration}_ifold{i_fold}.net', filetype='net')

                    dict_prediction_metrics_pemfp = self.calculatePredictionMetrics(bn=pemfp_bn,
                                                                                    dict_query_sets=dict_query_sets,
                                                                                    dict_evidence_sets=dict_evidence_sets,
                                                                                    dict_query_y=dict_query_y,
                                                                                    eval_metrics=eval_metrics,
                                                                                    backend="smile")

                    cur_place["pemfp"]["prediction_metrics_of_iteration"] = dict_prediction_metrics_pemfp
                    cur_place["pemfp"]["list_pemfp_calculated_log_likelihood_scores"] = list_pem_w_fp_calculated_log_likelihood_scores
                    cur_place["pemfp"]["list_pemfp_calculated_historic_cpds"] = list_pem_w_fp_calculated_historic_cpds
                    cur_place["pemfp"]["execution_time_time"] = delta_t_pemfp
                    cur_place["pemfp"]["execution_time_perf_counter"] = delta_t_pemfp_2

                    with open(f'./results_for_dissertation_find_the_fault_try_2024-12-02/dict_results_for_dissertation_cv_iteration_{cv_iteration}_ifold_{i_fold}.pickle', 'wb') as handle:
                        pickle.dump(dict_cv_results.copy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                except Exception as e:
                    print(f"Problem encountered. {e}")

        return dict_cv_results

    def extractQueryAndEvidenceSets(self, df_data):
        # prepare the query set and the evidence set of the df_test_data
        bdsn_inference_objectives = ["IZ1", "IZ2", "IZ3", "IZ4"]
        dict_evidence_sets = {}
        dict_query_sets = {}
        dict_query_y = {}
        for iz in bdsn_inference_objectives:
            if iz == "IZ1":
                # IZ1: Identifikation der wahrscheinlichsten Fehlerursache für einen identifizierten Fehler (diagnostisches Schließen)
                rel_evidence_cols = []
                rel_query_cols = []
                for col_name in df_data.columns:
                    if "Func" in col_name or "Komponente" in col_name:
                        rel_evidence_cols.append(col_name)
                    if "FC" in col_name:
                        rel_query_cols.append(col_name)

                rel_evidence_data = df_data[rel_evidence_cols].copy()
                cur_evidence_list = []
                for idx, row in rel_evidence_data.iterrows():
                    cur_ev_dict = {si: sv for si, sv in row.items() if sv != "*" and sv != np.nan}
                    if len(cur_ev_dict) == 0:
                        print()
                    cur_evidence_list.append(cur_ev_dict)

                rel_query_data = df_data[rel_query_cols].copy()
                cur_query_list = []
                cur_query_y_list = []
                for idx, row in rel_query_data.iterrows():
                    cur_query_list.append(rel_query_cols)
                    cur_query_y_dict = {si: sv for si, sv in row.items()}
                    cur_query_y_list.append(cur_query_y_dict)

                dict_evidence_sets[iz] = cur_evidence_list
                dict_query_sets[iz] = cur_query_list
                dict_query_y[iz] = cur_query_y_list

            elif iz == "IZ2":
                # IZ2: Identifikation des wahrscheinlichsten Fehlers und Fehlerekffektes für eine
                # identifizierte Fehlerursache (prädiktives Schließen).
                rel_evidence_cols = []
                rel_query_cols = []
                for col_name in df_data.columns:
                    if "FC" in col_name:
                        rel_evidence_cols.append(col_name)
                    if "Func" in col_name or "Komponente" in col_name or "FE" in col_name:
                        rel_query_cols.append(col_name)

                rel_evidence_data = df_data[rel_evidence_cols].copy()
                cur_evidence_list = []
                for idx, row in rel_evidence_data.iterrows():
                    cur_ev_dict = {si: sv for si, sv in row.items() if sv != "*" and sv != np.nan}
                    if len(cur_ev_dict) == 0:
                        print()
                    cur_evidence_list.append(cur_ev_dict)

                rel_query_data = df_data[rel_query_cols].copy()
                cur_query_list = []
                cur_query_y_list = []
                for idx, row in rel_query_data.iterrows():
                    cur_query_list.append(rel_query_cols)
                    cur_query_y_dict = {si: sv for si, sv in row.items()}
                    cur_query_y_list.append(cur_query_y_dict)

                dict_evidence_sets[iz] = cur_evidence_list
                dict_query_sets[iz] = cur_query_list
                dict_query_y[iz] = cur_query_y_list

            elif iz == "IZ3":
                # IZ3: Ableitung einer zielgerichteten Handlungsempfehlung, die am wahrscheinlichsten einen
                # identifizierten Fehler bzw. eine identifizierte Fehlerursache behebt (prädiktives Schließen)
                rel_evidence_cols_fc = []
                rel_query_cols_fc = []
                for col_name in df_data.columns:
                    if "FC" in col_name or "Func" in col_name or "Komponente" in col_name:
                        rel_evidence_cols_fc.append(col_name)
                    if "CM" in col_name:
                        rel_query_cols_fc.append(col_name)

                """
                rel_evidence_cols_f = []
                rel_query_cols_f = []
                for col_name in df_data.columns:
                    if "Func" in col_name or "Komponente" in col_name:
                        rel_evidence_cols_f.append(col_name)
                    if "CM" in col_name:
                        rel_query_cols_f.append(col_name)
                """

                # get evidence data
                rel_evidence_data = df_data[rel_evidence_cols_fc].copy()
                cur_evidence_list = []
                for idx, row in rel_evidence_data.iterrows():
                    cur_ev_dict = {si: sv for si, sv in row.items() if "FC" not in si and sv != "*" and sv != np.nan}
                    if len(cur_ev_dict) == 0:
                        print()
                    cur_evidence_list.append(cur_ev_dict)

                    cur_ev_dict = {si: sv for si, sv in row.items() if "Func" not in si and "Komponente" not in si and sv != "*" and sv != np.nan}
                    if len(cur_ev_dict) == 0:
                        print()
                    cur_evidence_list.append(cur_ev_dict)

                # get query data for fault causes
                rel_query_data = df_data[rel_query_cols_fc].copy()
                cur_query_list = []
                cur_query_y_list = []
                for idx, row in rel_query_data.iterrows():
                    cur_query_list.append(rel_query_cols_fc)
                    cur_query_list.append(rel_query_cols_fc)

                    cur_query_y_dict = {si: sv for si, sv in row.items()}
                    cur_query_y_list.append(cur_query_y_dict)
                    cur_query_y_list.append(cur_query_y_dict)

                dict_evidence_sets[iz] = cur_evidence_list
                dict_query_sets[iz] = cur_query_list
                dict_query_y[iz] = cur_query_y_list

            elif iz == "IZ4":
                # IZ4: Identifikation des wahrscheinlichsten Fehlers und der wahrscheinlichsten Fehlerursache für einen
                # identifizierten Fehlereffekt (diagnostisches Schließen)
                rel_evidence_cols = []
                rel_query_cols = []
                for col_name in df_data.columns:
                    if "FE" in col_name:
                        rel_evidence_cols.append(col_name)
                    if "Func" in col_name or "Komponente" in col_name or "FC" in col_name:
                        rel_query_cols.append(col_name)

                rel_evidence_data = df_data[rel_evidence_cols].copy()
                cur_evidence_list = []
                for idx, row in rel_evidence_data.iterrows():
                    cur_ev_dict = {si: sv for si, sv in row.items() if sv != "*" and sv != np.nan}
                    if len(cur_ev_dict) == 0:
                        print()
                    cur_evidence_list.append(cur_ev_dict)

                rel_query_data = df_data[rel_query_cols].copy()
                cur_query_list = []
                cur_query_y_list = []
                for idx, row in rel_query_data.iterrows():
                    cur_query_list.append(rel_query_cols)
                    cur_query_y_dict = {si: sv for si, sv in row.items()}
                    cur_query_y_list.append(cur_query_y_dict)

                dict_evidence_sets[iz] = cur_evidence_list
                dict_query_sets[iz] = cur_query_list
                dict_query_y[iz] = cur_query_y_list

        return dict_query_sets, dict_evidence_sets, dict_query_y

    def calculatePredictionMetrics(self, bn, dict_query_sets, dict_evidence_sets, dict_query_y, eval_metrics, backend="pgmpy"):
        if backend == "pgmpy":
            cia = VariableElimination(bn.copy())
        elif backend == "smile":
            bn.save('./tmp_bn/tmp_bn_for_evaluation.net', filetype='net')
            pysmile_bn = pysmile.Network()
            pysmile_bn.read_file("./tmp_bn/tmp_pem_bn.net")
        else:
            raise ValueError("Backend unknown!")

        dict_prediction_metrics = dict()

        # iterate all inference targets
        for iz in dict_evidence_sets.keys():
            print("============================================")
            print(f"IZ: {iz}")
            dict_prediction_metrics[iz] = dict()
            list_iz_evidence_set = dict_evidence_sets[iz]
            list_iz_query_set = dict_query_sets[iz]
            list_iz_query_y = dict_query_y[iz]

            list_all_query_nodes = []
            for idx in range(len(list_iz_query_set)):
                for idx2 in range(len(list_iz_query_set[idx])):
                    if list_iz_query_set[idx][idx2] not in list_all_query_nodes:
                        list_all_query_nodes.append(list_iz_query_set[idx][idx2])

            dict_all_query_nodes_predictions = dict()
            dict_all_query_nodes_targets = dict()
            for nidx in list_all_query_nodes:
                num_samples = len(list_iz_query_set)
                num_states = bn.get_cpds(nidx).cardinality[0]
                dict_all_query_nodes_predictions[nidx] = np.full(shape=(num_samples, num_states), fill_value=np.nan)
                dict_all_query_nodes_targets[nidx] = np.full(shape=(num_samples, num_states), fill_value=np.nan)

            for idx in range(len(list_iz_evidence_set)):
                dict_cur_evidence = list_iz_evidence_set[idx]
                list_cur_query = list_iz_query_set[idx]
                dict_cur_query_y = list_iz_query_y[idx]

                #if len(dict_cur_evidence) == 0:
                #    qr = None
                #else:
                # query the Bayesian network
                print(f"Query idx: {idx} with backend {backend}")

                if backend == "pgmpy":
                    qr = cia.query(variables=list_cur_query, evidence=dict_cur_evidence, joint=False, elimination_order="greedy")
                elif backend == "smile":
                    pysmile_bn.clear_all_evidence()
                    for ek, ev in dict_cur_evidence.items():
                        pysmile_bn.set_evidence(ek, ev)
                    pysmile_bn.update_beliefs()

                    qr_smile = {qle.variable: qle for qle in bn.get_cpds() if qle.variable in list_cur_query}.copy()
                    for qk, qv in dict_cur_query_y.items():
                        if pysmile_bn.is_evidence(qk):
                            print(qk + " has evidence set (" + pysmile_bn.get_outcome_id(qk, pysmile_bn.get_evidence(qk)) + ")")
                            qr_smile[qk] = pysmile_bn.get_outcome_id(qk, pysmile_bn.get_evidence(qk))
                        else:
                            posteriors = pysmile_bn.get_node_value(qk)
                            qr_smile[qk].values = np.asarray(posteriors)

                for dk, dv in dict_cur_query_y.items():
                    if len(dict_cur_evidence) > 0:
                        cur_prediction_array = qr[dk].values if backend == "pgmpy" else qr_smile[dk].values
                        cur_target_array = np.full(shape=cur_prediction_array.shape, fill_value=np.nan)
                    else:
                        #cur_prediction_array = np.full(shape=bn.get_cpds(dk).values.shape, fill_value=np.nan)
                        cur_prediction_array = np.full(shape=qr[dk].values.shape if backend == "pgmpy" else qr_smile[dk].values.shape, fill_value=np.nan)
                        cur_target_array = np.full(shape=cur_prediction_array.shape, fill_value=np.nan)

                    cur_target = dict_cur_query_y[dk]
                    if cur_target == "*":
                        dict_all_query_nodes_targets[dk][idx, :] = cur_target_array
                        dict_all_query_nodes_predictions[dk][idx, :] = cur_prediction_array
                    else:
                        cur_target_array[:] = 0
                        idx_pos = qr[dk].name_to_no[dk][cur_target] if backend == "pgmpy" else qr_smile[dk].name_to_no[dk][cur_target]
                        #idx_pos_2 = dict_query_y[dk].name_to_no[dk][cur_target]
                        cur_target_array[idx_pos] = 1.0

                        dict_all_query_nodes_targets[dk][idx, :] = cur_target_array
                        dict_all_query_nodes_predictions[dk][idx, :] = cur_prediction_array

            # calculate the metrics for each inference target and each node in the BN
            dict_iz_eval_results = dict()
            for current_node in dict_all_query_nodes_predictions.keys():
                cur_predictions = dict_all_query_nodes_predictions[current_node]
                cur_targets = dict_all_query_nodes_targets[current_node]

                dict_iz_eval_results[current_node] = dict()
                for mn, evmcs in eval_metrics.items():
                    result, ccounts = evmcs(predictions=cur_predictions, y=cur_targets)
                    dict_iz_eval_results[current_node][mn] = result
                    dict_iz_eval_results[current_node][f"{mn}_counts"] = ccounts

            # calculate means over the nodes
            dict_prediction_metrics[iz]["dict_iz_eval_results"] = dict_iz_eval_results
            for mk in eval_metrics.keys():
                cur_metric_vals = [cv[mk] for ck, cv in dict_iz_eval_results.items()]
                cur_counts_vals = [cv[f"{mk}_counts"] for ck, cv in dict_iz_eval_results.items()]
                dict_prediction_metrics[iz][mk] = np.mean(cur_metric_vals)
                dict_prediction_metrics[iz][f"{mk}_counts"] = np.sum(cur_counts_vals)

        # calcualte means over the inference targets
        dict_list_of_metrics = {k: [] for k, _ in eval_metrics.items()}
        for kiz, viz in dict_prediction_metrics.items():
            for mk, _ in eval_metrics.items():
                dict_list_of_metrics[mk].append(viz[mk])

        for klom, vlom in dict_list_of_metrics.items():
            dict_prediction_metrics[f"overall_{klom}"] = np.mean(vlom)

        return dict_prediction_metrics

    def predictionAccuracy(self, predictions, y):
        # delete nan rows
        ix, iy = np.where(np.isnan(y))
        unique_rows = np.unique(ix)
        y = np.delete(y, tuple(unique_rows), axis=0)
        predictions = np.delete(predictions, tuple(unique_rows), axis=0)

        ix, iy = np.where(np.isnan(predictions))
        unique_rows = np.unique(ix)
        y = np.delete(y, tuple(unique_rows), axis=0)
        predictions = np.delete(predictions, tuple(unique_rows), axis=0)

        #calculate the accuracy score
        arg_max_predictions = np.argmax(predictions, axis=1)
        arg_max_y = np.argmax(y, axis=1)
        acc_score = accuracy_score(y_true=arg_max_y, y_pred=arg_max_predictions)
        ccounts = arg_max_predictions.shape[0]
        return acc_score, ccounts

    def predictionBrierScore(self, predictions, y):
        """
        Calculate the brier score for multi-class problems according to Brier 1950
        :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
        :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
        :return: The brier score.
        """
        # delete nan rows
        ix, iy = np.where(np.isnan(y))
        unique_rows = np.unique(ix)
        y = np.delete(y, tuple(unique_rows), axis=0)
        predictions = np.delete(predictions, tuple(unique_rows), axis=0)

        ix, iy = np.where(np.isnan(predictions))
        unique_rows = np.unique(ix)
        y = np.delete(y, tuple(unique_rows), axis=0)
        predictions = np.delete(predictions, tuple(unique_rows), axis=0)

        # calculate the Brier score
        y_true = y
        y_pred = predictions
        if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
            raise ValueError(
                "y_true and y_pred need to be 2D and have the same shape, "
                "got {} and {} instead.".format(y_true.shape, y_pred.shape)
            )

        result = np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
        ccounts = y_pred.shape[0]
        return result, ccounts

    def bnLogLikelihood(self, predictions, y):
        return 0

    def getDatasetDescription(self, dataset=None):
        rstr = ""
        num_total_values = dataset.shape[0] * dataset.shape[1]
        num_missing = 0
        rstr += "Number of total values in dataset: " + str(num_total_values) + "\n"
        for col in dataset.columns:
            c = dataset[col].value_counts()
            if "*" in c.index:
                num_missing += c["*"]

        rstr += "Number of missing values in dataset: " + str(num_missing) + "\n"
        rstr += "Proportion of missing values in dataset: " + str(num_missing/num_total_values * 100) + "%"
        return rstr

    def getResultsForDissertation(self, dict_cv_results):
        dict_results_for_dissertation = {}
        #list_algorithms = ["em", "pem", "pemfp"]
        list_algorithms = ["em", "pemfp"]
        list_metrics = ["accuracy", "brier_score"]
        list_izs = ["IZ1", "IZ2", "IZ3", "IZ4"]

        for alg in list_algorithms:
            dict_results_for_dissertation[alg] = dict()
            dict_results_for_dissertation[alg]["execution_time_time"] = []
            dict_results_for_dissertation[alg]["execution_time_perf_counter"] = []
            dict_results_for_dissertation[alg]["log_likelihood"] = []
            dict_results_for_dissertation[alg][f"list_{alg}_num_itr"] = []
            for metrc in list_metrics:
                dict_results_for_dissertation[alg][f"total_overall_{metrc}"] = 0
                dict_results_for_dissertation[alg][f"std_overall_{metrc}"] = 0
                dict_results_for_dissertation[alg][f"overall_{metrc}_of_iteration"] = []

            for iz in list_izs:
                dict_results_for_dissertation[alg][iz] = dict()
                for mtrc in list_metrics:
                    dict_results_for_dissertation[alg][iz][mtrc] = []
                    dict_results_for_dissertation[alg][iz][f"total_{mtrc}"] = 0
                    dict_results_for_dissertation[alg][iz][f"std_{mtrc}"] = 0

        for cv_iteration, cv_dict in dict_cv_results.items():
            for i_fold, fold_dict in cv_dict.items():
                for k_algorithm, algo_dict in fold_dict.items():
                    if len(algo_dict) == 0:
                        continue
                    execution_time_perf_counter = algo_dict["execution_time_perf_counter"]
                    dict_results_for_dissertation[k_algorithm]["execution_time_time"].append(execution_time_perf_counter)

                    execution_time_time = algo_dict["execution_time_time"]
                    dict_results_for_dissertation[k_algorithm]["execution_time_perf_counter"].append(execution_time_time)

                    log_likelihood = algo_dict[f"list_{k_algorithm}_calculated_log_likelihood_scores"][-1]
                    num_alg_itr = len(algo_dict[f"list_{k_algorithm}_calculated_log_likelihood_scores"])
                    dict_results_for_dissertation[k_algorithm]["log_likelihood"].append(log_likelihood)
                    dict_results_for_dissertation[k_algorithm][f"list_{k_algorithm}_num_itr"].append(num_alg_itr)

                    for iz, iz_dict in algo_dict["prediction_metrics_of_iteration"].items():
                        if "IZ" in iz:
                            for mtrc in list_metrics:
                                dict_results_for_dissertation[k_algorithm][iz][mtrc].append(iz_dict[mtrc])
                        if "overall_accuracy" in iz:
                            dict_results_for_dissertation[k_algorithm]["overall_accuracy_of_iteration"].append(iz_dict)
                        if "overall_brier_score" in iz:
                            dict_results_for_dissertation[k_algorithm]["overall_brier_score_of_iteration"].append(iz_dict)

        # calculate means and stds
        for algo_name, algo_dict in dict_results_for_dissertation.items():
            algo_dict["total_mean_execution_time_time"] = np.mean(algo_dict["execution_time_time"])
            algo_dict["total_std_execution_time_time"] = np.std(algo_dict["execution_time_time"])
            algo_dict["2.5th-97.5th_execution_time_time"] = np.percentile(algo_dict["execution_time_time"], [2.5, 97.5])

            algo_dict["total_mean_execution_time_perf_counter"] = np.mean(algo_dict["execution_time_perf_counter"])
            algo_dict["total_std_execution_time_perf_counter"] = np.std(algo_dict["execution_time_perf_counter"])
            algo_dict["2.5th-97.5th_execution_time_perf_counter"] = np.percentile(algo_dict["execution_time_perf_counter"], [2.5, 97.5])

            algo_dict["total_overall_accuracy"] = np.mean(algo_dict['overall_accuracy_of_iteration'])
            algo_dict["std_overall_accuracy"] = np.std(algo_dict['overall_accuracy_of_iteration'])
            algo_dict["2.5th-97.5th_accuracy"] = np.percentile(algo_dict["overall_accuracy_of_iteration"], [2.5, 97.5])

            algo_dict["total_overall_brier_score"] = np.mean(algo_dict['overall_brier_score_of_iteration'])
            algo_dict["std_overall_brier_score"] = np.std(algo_dict["overall_brier_score_of_iteration"])
            algo_dict["2.5th-97.5th_brier_score"] = np.percentile(algo_dict["overall_brier_score_of_iteration"], [2.5, 97.5])

            # mean and std of metrics for each IZ
            for iz in list_izs:
                algo_dict[iz]["total_accuracy"] = np.mean(algo_dict[iz]["accuracy"])
                algo_dict[iz]["std_accuracy"] = np.std(algo_dict[iz]["accuracy"])
                algo_dict[iz]["2.5th-97.5th_accuracy"] = np.percentile(algo_dict[iz]["accuracy"], [2.5, 97.5])

                algo_dict[iz]["total_brier_score"] = np.mean(algo_dict[iz]["brier_score"])
                algo_dict[iz]["std_brier_score"] = np.std(algo_dict[iz]["brier_score"])
                algo_dict[iz]["2.5th-97.5th_brier_score"] = np.percentile(algo_dict[iz]["brier_score"], [2.5, 97.5])

        return dict_results_for_dissertation