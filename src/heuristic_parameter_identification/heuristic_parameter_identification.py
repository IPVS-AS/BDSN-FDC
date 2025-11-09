import typing
import logging
import argparse
import pandas as pd
import numpy as np
from time import perf_counter_ns
from scipy.stats import bootstrap
import re

from bdsn_bayesian_network import BdsnBayesianNetwork
from bdsn_pgmpy_bayesian_network import PgmpyBdsnBayesianNetwork

import pysmile_license # pysmile license required in the same directory of this python script
import pysmile

def get_occurrence_probability(o):
    """
    Get the probability value associated with the FMECA occurrence value.
    :param o: Occurence Probability O according to FMECA
    :return: Probability value for O according to the AIAG & VDA FMEA handbook
    """
    x0 = np.arange(1, 11, 1)
    y = np.asarray([0, 0.001 / 1000, 0.01 / 1000, 0.1 / 1000, 0.5 / 1000, 2 / 1000, 10 / 1000, 20 / 1000, 50 / 1000, 100 / 1000])
    return y[np.where(x0 == o)][0]

def read_in_fmeca_data(path: str):
    """
    Function to read in the fmeca data given in the `csv` file format.
    :param path: str path to file
    :return: `pandas.DataFrame` of the imported csv data
    """
    df_fmeca_data = pd.read_csv(path, sep=";")
    return df_fmeca_data

def get_nested_value_safe(data_dict, path, delimiter="/", default=None):
    keys = path.split(delimiter)
    current = data_dict
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current

def heuristic_parameter_identification(path : str, logger, verbose : int =0):
    """
    Function for the heuristic parameter identification approach based on FMECA data
    :return:
    """
    t_start_read_in_fmeca_data = perf_counter_ns() * 10 ** -6
    df_fmeca_data = read_in_fmeca_data(path=path)
    t_end_read_in_fmeca_data = perf_counter_ns() * 10 ** -6
    delta_t_read_in_fmeca_data = t_end_read_in_fmeca_data - t_start_read_in_fmeca_data
    logger.debug(f"Successfully imported FMECA data within {delta_t_read_in_fmeca_data:.2f} ms.")
    logger.debug(df_fmeca_data)

    '''
    Create the BN structure
    '''
    t_start_creation_bn_structure = perf_counter_ns() * 10**-6

    # look up dict for fault root causes
    dict_look_up_fault_root_causes = {}
    df_fc_f = df_fmeca_data.groupby(by="Fault_Root_Cause")["Function"].apply(list).reset_index(name="affected_functions")
    df_f_fm = df_fmeca_data.groupby(by="Function")["Failure_Mode"].apply(list).reset_index(name="failure_modes_of_function")
    df_fc_fm = df_fmeca_data.groupby(by="Fault_Root_Cause")["Failure_Mode"].apply(list).reset_index(name="caused_failure_modes")
    df_fc_f_fm = df_fmeca_data.groupby(by=["Fault_Root_Cause", "Function"])["Failure_Mode"].unique().apply(list).reset_index(name="caused_failure_modes")
    df_fc_f_o_fm = df_fmeca_data.groupby(by=["Fault_Root_Cause", "Function", "Occurrence_Probability_O"])["Failure_Mode"].unique().apply(list).reset_index(name="caused_failure_modes")
    df_fc_f_fm_o = df_fmeca_data.groupby(by=["Fault_Root_Cause", "Function", "Failure_Mode", "Occurrence_Probability_O"])["Failure_Mode", "Fault_Root_Cause"]
    #df_fc_f_fm_table = df_fmeca_data.groupby(by=["Fault_Root_Cause", "Function"])["Failure_Mode"].unique().reset_index(name="caused_failure_modes")
    df_fc_fm_number = df_fmeca_data.groupby(by="Fault_Root_Cause")["Failure_Mode"].count().reset_index(name="number_of_caused_failure_modes")

    # look up dict for fault effects
    dict_look_up_fault_effects = {}
    dict_look_up_s_max = {}
    df_fm_fe = df_fmeca_data.groupby(by="Failure_Mode")["Fault_Effect"].unique().apply(list).reset_index(name="evoked_fault_effects")
    df_f_fm_fe = df_fmeca_data.groupby(by=["Function", "Failure_Mode"])["Fault_Effect"].unique().apply(list).reset_index(name="evoked_fault_effects")
    df_fe_s = df_fmeca_data.groupby(by=["Fault_Effect"])["Severity_S"].unique().apply(list).reset_index(name="List_S")
    df_fe_s["S"] = df_fe_s['List_S'].apply(lambda x: np.max(x))

    # look up for causal chains
    expression_fe = r'^FE[\d][\d]_(.*)'
    expression_fc = r'^FC[\d][\d]_(.*)'
    df_fe = df_fmeca_data[['Fault_Effect', 'Function']].reset_index()
    df_fe["FE_text"] = df_fe["Fault_Effect"].str.extract(expression_fe)
    df_fe = df_fe.rename(columns={"Function": "Causing_Function"})

    df_fc = df_fmeca_data[["Fault_Root_Cause", "Function"]].reset_index()
    df_fc["FC_text"] = df_fc['Fault_Root_Cause'].str.extract(expression_fc)
    df_fc =df_fc.rename(columns={"Function": "Affected_Function"})

    df_fe_fc_joined = pd.merge(df_fe, df_fc, how='inner', left_on=['FE_text'], right_on=['FC_text'])
    df_fe_fc_joined["causal_chain"] = df_fe_fc_joined["Fault_Root_Cause"].str.extract(r'^(FC[\d][\d]_)') + df_fe_fc_joined["Fault_Effect"]

    '''
    1. Build tree structure of BN
    '''
    # Get the system hierarchy
    dict_bn_tree = dict()

    system_elements = df_fmeca_data[["System_Element", "Functional_Unit", "Function"]]
    for i, r in system_elements.iterrows():
        hierarchy = r["System_Element"].split("-")
        for j, he in enumerate(hierarchy):
            if j == 0 and he not in dict_bn_tree.keys():
                dict_bn_tree[hierarchy[0]] = {}
                continue

            if j == 1 and he not in dict_bn_tree[hierarchy[0]].keys():
                dict_bn_tree[hierarchy[0]][hierarchy[1]] = {}
                continue

            if j == 2 and he not in dict_bn_tree[hierarchy[0]][hierarchy[1]].keys():
                dict_bn_tree[hierarchy[0]][hierarchy[1]][hierarchy[2]] = {}
                continue

        if r["Functional_Unit"] not in get_nested_value_safe(data_dict=dict_bn_tree, path="/".join(hierarchy)):
            get_nested_value_safe(data_dict=dict_bn_tree, path="/".join(hierarchy))[r["Functional_Unit"]] = {}

        if r["Function"] not in get_nested_value_safe(data_dict=dict_bn_tree, path="/".join(hierarchy)):
            get_nested_value_safe(data_dict=dict_bn_tree, path="/".join(hierarchy))[r["Functional_Unit"]][r["Function"]] = {}

    """
    Get Failure Modes for each functional unit, i. e., functional unit is either a component, process or function
    """

    def get_failable_unit(cur_dict):
        copy_dict = cur_dict.copy()
        ret_list = []
        if type(copy_dict) == dict and len(copy_dict) > 0:
            for k, v in copy_dict.items():
                if type(v) == dict and len(v) > 0:
                    ret_list = ret_list + get_failable_unit(v)
                elif type(v) == dict and len(v) == 0:
                    ret_list.append(k)
        # elif type(copy_dict) == dict and len(copy_dict) == 0:
        #    ret_list.append()
        return ret_list

    # get failable units
    list_failable_units = get_failable_unit(dict_bn_tree)

    # Get for each failable unit the unique failure modes
    dict_failable_units = dict()
    for failable_unit in list_failable_units:
        dict_failable_units[failable_unit] = df_fmeca_data[df_fmeca_data["Function"] == failable_unit]["Failure_Mode"].unique()

    # Get FAILURE CAUSE NODES
    list_names_failure_cause_nodes = df_fmeca_data['Fault_Root_Cause'].unique()

    # Get FAILURE EFFECT NODES
    list_names_failure_effect_nodes = df_fmeca_data['Fault_Effect'].unique()

    # Get Corrective Measures Nodes
    list_names_corrective_measure_nodes = df_fmeca_data["Corrective_Measure"].unique()

    # Create Bayesian Network
    bdsn_bayesian_network = BdsnBayesianNetwork(logger=logger)
    bdsn_pgmpy_bayesian_network = PgmpyBdsnBayesianNetwork(logger=logger)

    # Add the structural component nodes
    def addStructuralElementsToBayesianNetwork(cur_dict, i_x_level, i_y_level, x_pos_0, y_pos_0, node_width,
                                               node_height, handle_child_node):
        copy_dict = cur_dict.copy()
        i_x_level_0 = i_x_level
        set_structural_nodes = set()
        if type(copy_dict) == dict and len(copy_dict) > 0:
            for i, (k, v) in enumerate(copy_dict.items()):
                if type(v) == dict and len(v) > 0:
                    id_created_node = bdsn_bayesian_network.createDiscreteDeterministicNode(id=k,
                                                                                            name=k,
                                                                                            bdsn_node_type=BdsnBayesianNetwork.STRUCTURAL_COMPONENT_NODE if re.match(r"^SE[\d][\d]_(.*)$", k) else BdsnBayesianNetwork.FUNCTIONAL_UNIT_NODE,
                                                                                            node_states=["OK", "NOK"],
                                                                                            x_pos=x_pos_0 + i_x_level * (
                                                                                                    node_width + 40),
                                                                                            y_pos=y_pos_0 - i_y_level * (
                                                                                                    node_height + 80),
                                                                                            width=node_width,
                                                                                            height=node_height)
                    set_structural_nodes.add(k)

                    if handle_child_node is not None:
                        bdsn_bayesian_network.addArc(handle_node_parent=id_created_node,
                                                     handle_node_child=handle_child_node)

                    i_x_level, i_y_level, ret_set = addStructuralElementsToBayesianNetwork(v,
                                                                                           i_x_level=i_x_level,
                                                                                           i_y_level=i_y_level + 1,
                                                                                           x_pos_0=x_pos_0,
                                                                                           y_pos_0=y_pos_0,
                                                                                           node_width=node_width,
                                                                                           node_height=node_height,
                                                                                           handle_child_node=id_created_node)
                    set_structural_nodes = set_structural_nodes.union(ret_set)

            return i_x_level + 1, i_y_level - 1, set_structural_nodes

    x_pos_0 = 500
    y_pos_0 = 1200
    node_width = 100
    node_height = 75
    _, _, set_structural_nodes = addStructuralElementsToBayesianNetwork(dict_bn_tree,
                                                                        i_x_level=0,
                                                                        i_y_level=0,
                                                                        x_pos_0=x_pos_0,
                                                                        y_pos_0=y_pos_0,
                                                                        node_height=node_height,
                                                                        node_width=node_width,
                                                                        handle_child_node=None)

    for structural_node in set_structural_nodes:
        bdsn_bayesian_network.initializeDiscreteStructuralElementNodes(structural_node)

    # Add all root nodes of unique Failure Cause nodes to the network
    x_pos_0 = 40
    y_pos_0 = 200
    node_width = 100
    node_height = 75
    for i, name_failure_cause_node in enumerate(list_names_failure_cause_nodes):
        # do not add nodes part of causal chains
        if name_failure_cause_node in df_fe_fc_joined["Fault_Root_Cause"].to_list():
            continue

        id_created_node = bdsn_bayesian_network.createCptNode(id=name_failure_cause_node,
                                                              name=name_failure_cause_node,
                                                              bdsn_node_type=BdsnBayesianNetwork.FAILURE_CAUSE_NODE,
                                                              node_states=["True", "False"],
                                                              x_pos=x_pos_0 + i * (node_width + x_pos_0),
                                                              y_pos=y_pos_0,
                                                              width=node_width,
                                                              height=node_height)

        # pgmpy
        bdsn_pgmpy_bayesian_network.add_node(name_failure_cause_node)
        cur_cpd = bdsn_pgmpy_bayesian_network.defineCPD(variable=name_failure_cause_node,
                                                        variable_card=2,
                                                        values=[[0.2], [0.8]],
                                                        state_names={name_failure_cause_node: ["True", "False"]})
        bdsn_pgmpy_bayesian_network.add_cpds(cur_cpd)

    # Add all effect nodes of unique Failure Effects to the network
    x_pos_0 = 120
    y_pos_0 = 600
    node_width = 100
    node_height = 75
    for i, name_failure_effect_node in enumerate(list_names_failure_effect_nodes):
        # do not add nodes being part of causal chains
        if name_failure_effect_node in df_fe_fc_joined["Fault_Effect"].to_list():
            continue

        id_created_node = bdsn_bayesian_network.createCptNode(id=name_failure_effect_node,
                                                              name=name_failure_effect_node,
                                                              bdsn_node_type=BdsnBayesianNetwork.FAILURE_EFFECT_NODE,
                                                              node_states=["True", "False"],
                                                              x_pos=x_pos_0 + i * (node_width + x_pos_0),
                                                              y_pos=y_pos_0,
                                                              width=node_width,
                                                              height=node_height)

        # pgmpy
        bdsn_pgmpy_bayesian_network.add_node(name_failure_effect_node)

    # Add all nodes representing causal chains to the network
    x_pos_0 = 180
    y_pos_0 = 475
    node_width = 100
    node_height = 75
    for i, name_causal_chain_node in enumerate(df_fe_fc_joined["causal_chain"].to_list()):
        id_created_node = bdsn_bayesian_network.createCptNode(id=name_causal_chain_node,
                                                              name=name_causal_chain_node,
                                                              bdsn_node_type=BdsnBayesianNetwork.CAUSAL_CHAIN_NODE,
                                                              node_states=["True", "False"],
                                                              x_pos=x_pos_0 + i * (node_width + x_pos_0),
                                                              y_pos=y_pos_0,
                                                              width=node_width,
                                                              height=node_height)

        # pgmpy
        bdsn_pgmpy_bayesian_network.add_node(name_causal_chain_node)

    # Add all function nodes with failure modes as states
    def getStructuralChildNode(dict_bn_tree, key_to_search):
        for k, v in dict_bn_tree.items():
            if key_to_search in v.keys():
                return k
            else:
                if type(v) == dict and len(v) > 0:
                    ret_val = getStructuralChildNode(v, key_to_search)

                    if ret_val != "None":
                        return ret_val

        return "None"

    x_pos_0 = 120
    y_pos_0 = 300
    node_width = 100
    node_height_per_state = 35
    set_unique_functional_units = set()
    for i, (k, v) in enumerate(dict_failable_units.items()):
        node_states = np.concatenate([v, ["UNK", "OK"]])
        id_created_node = bdsn_bayesian_network.createCptNode(id=k,
                                                              name=k,
                                                              bdsn_node_type=BdsnBayesianNetwork.FUNCTION_NODE,
                                                              node_states=node_states,
                                                              x_pos=x_pos_0 + i * (node_width + x_pos_0),
                                                              y_pos=y_pos_0,
                                                              width=node_width,
                                                              height=node_height_per_state * len(node_states))

        # pgmpy
        bdsn_pgmpy_bayesian_network.add_node(k)

        # arc to structural element
        child_node = getStructuralChildNode(dict_bn_tree, k)
        #print(k)
        #print(child_node)
        if child_node is not None:
            bdsn_bayesian_network.addArc(handle_node_parent=bdsn_bayesian_network.getNodeHandle(k),
                                         handle_node_child=bdsn_bayesian_network.getNodeHandle(child_node))
            set_unique_functional_units.add(child_node)
        else:
            print("Why?")

        relevant_nodes = df_fmeca_data[df_fmeca_data['Function'] == k]['Fault_Effect'].unique()

        # arc to effects
        if "FU" in k:
            relevant_nodes = df_fmeca_data[df_fmeca_data['Functional_Unit'] == k]['Fault_Effect'].unique()
        # elif "F0" in k:
        #     relevant_nodes = df_fmeca_data[df_fmeca_data['Function'] == k]['Fault_Effect'].unique()

        relevant_nodes = np.unique(relevant_nodes)
        if len(relevant_nodes) > 0:
            for rel_node in relevant_nodes:
                if rel_node != "None" and rel_node != "*":
                    if rel_node in df_fe_fc_joined["Fault_Effect"].to_list():
                        rel_causal_node = df_fe_fc_joined[(df_fe_fc_joined["Fault_Effect"] == rel_node) & (df_fe_fc_joined["Causing_Function"] == k)]["causal_chain"][0]
                        bdsn_bayesian_network.addArc(handle_node_parent=bdsn_bayesian_network.getNodeHandle(k),
                                                     handle_node_child=bdsn_bayesian_network.getNodeHandle(rel_causal_node))
                    else:
                        bdsn_bayesian_network.addArc(handle_node_parent=bdsn_bayesian_network.getNodeHandle(k),
                                                     handle_node_child=bdsn_bayesian_network.getNodeHandle(rel_node))

                    # pgmpy
                    bdsn_pgmpy_bayesian_network.add_edge(u=k, v=rel_node)



        # arc from cause to function nodes
        relevant_cause_nodes = df_fmeca_data[df_fmeca_data['Function'] == k]['Fault_Root_Cause'].unique()

        if "FU" in k:
            relevant_cause_nodes = df_fmeca_data[df_fmeca_data['Functional_Unit'] == k]['Fault_Root_Cause'].unique()

        relevant_cause_nodes = np.unique(relevant_cause_nodes)
        if len(relevant_cause_nodes) > 0:
            for rel_node in relevant_cause_nodes:
                if rel_node != "None" and rel_node != "*":
                    if rel_node in df_fe_fc_joined["Fault_Root_Cause"].to_list():
                        rel_causal_node = df_fe_fc_joined[(df_fe_fc_joined["Fault_Root_Cause"] == rel_node) & (df_fe_fc_joined["Affected_Function"] == k)]["causal_chain"][0]
                        rel_affected_function_node = df_fe_fc_joined[(df_fe_fc_joined["Fault_Root_Cause"] == rel_node) & (df_fe_fc_joined["Affected_Function"] == k)]["Affected_Function"][0]
                        bdsn_bayesian_network.addArc(handle_node_parent=bdsn_bayesian_network.getNodeHandle(rel_causal_node),
                                                     handle_node_child=bdsn_bayesian_network.getNodeHandle(rel_affected_function_node))
                    else:
                        bdsn_bayesian_network.addArc(handle_node_parent=bdsn_bayesian_network.getNodeHandle(rel_node),
                                                     handle_node_child=bdsn_bayesian_network.getNodeHandle(k))

                    # pgmpy
                    bdsn_pgmpy_bayesian_network.add_edge(u=rel_node, v=k)

    # Add all correction measure nodes - Binary-State Nodes to represent multi-label predictions
    x_pos_0 = 40
    y_pos_0 = 10
    node_width = 100
    node_height = 75

    for i, corrective_measure_name in enumerate(list_names_corrective_measure_nodes):
        id_created_node = bdsn_bayesian_network.createCptNode(id=corrective_measure_name,
                                                              name=corrective_measure_name,
                                                              bdsn_node_type=BdsnBayesianNetwork.MEAUSRE_NODE,
                                                              node_states=["True", "False"],
                                                              x_pos=x_pos_0 + i * (node_width + x_pos_0),
                                                              y_pos=y_pos_0,
                                                              width=node_width,
                                                              height=node_height)

        # pgmpy
        bdsn_pgmpy_bayesian_network.add_node(corrective_measure_name)

        list_root = df_fmeca_data[df_fmeca_data["Corrective_Measure"] == corrective_measure_name]['Fault_Root_Cause'].unique()
        list_root = np.unique(list_root)
        list_root = np.delete(list_root, np.argwhere(list_root == "None"))
        list_root = np.delete(list_root, np.argwhere(list_root == "*"))

        # add the arc from root cause node to the measure node
        for k in list_root:
            if k not in df_fe_fc_joined["Fault_Root_Cause"].to_list():
                bdsn_bayesian_network.addArc(handle_node_parent=bdsn_bayesian_network.getNodeHandle(k),
                                             handle_node_child=id_created_node)
            else:
                rel_causal_node = df_fe_fc_joined[df_fe_fc_joined["Fault_Root_Cause"] == k]["causal_chain"].unique()[0]
                bdsn_bayesian_network.addArc(handle_node_parent=bdsn_bayesian_network.getNodeHandle(rel_causal_node),
                                             handle_node_child=id_created_node)

            # pgmpy
            bdsn_pgmpy_bayesian_network.add_edge(u=k, v=corrective_measure_name)

    # set states of deterministic node
    for i, set_element in enumerate(set_unique_functional_units):
        cur_cpt = bdsn_bayesian_network.getBayesianNetwork().get_node_definition(child_node)
        list_child_cpt_states = bdsn_bayesian_network.getListCptStates(child_node)
        parent_nodes = bdsn_bayesian_network.getBayesianNetwork().get_parents(child_node)
        dict_parent_states = {k: bdsn_bayesian_network.getListCptStates(k) for k in parent_nodes}

    # stop performance measurement
    t_end_creation_bn_structure = perf_counter_ns() * 10 ** -6
    delta_t_creation_bn_structure = t_end_creation_bn_structure - t_start_creation_bn_structure

    # bdsn_pgmpy_bayesian_network.printDAG()
    print(bdsn_pgmpy_bayesian_network.getBayesianNetwork().nodes())

    '''
    Heurisitic Parameter identification
    '''
    def indexToCoords(index, dim_sizes, coords):
        prod = 1
        for i in range(len(dim_sizes) - 1, -1, -1):
            coords[i] = int(index / prod) % dim_sizes[i]
            prod *= dim_sizes[i]

    t_start_heuristic_parameter_identification = perf_counter_ns() * 10**-6

    h = bdsn_bayesian_network.getBayesianNetwork().get_first_node()
    while (h >= 0):

        current_node_id = bdsn_bayesian_network.getBayesianNetwork().get_node_id(h)
        current_node_handle = bdsn_bayesian_network.getNodeHandle(current_node_id)
        list_parent_node_ids = bdsn_bayesian_network.getBayesianNetwork().get_parent_ids(h)
        list_parent_node_handles = bdsn_bayesian_network.getBayesianNetwork().get_parents(h)

        cpt = bdsn_bayesian_network.getBayesianNetwork().get_node_definition(current_node_handle)

        if re.match(r"^FC[\d][\d]_(?!FE[\d][\d]_)(.*)$", current_node_id):
            # 1) A priori probabilities of fault root cause nodes
            fc_os = df_fmeca_data[df_fmeca_data["Fault_Root_Cause"] == current_node_id][["Fault_Root_Cause", "Occurrence_Probability_O"]]
            unique_os = fc_os["Occurrence_Probability_O"].unique()
            if len(unique_os) > 1:
                logger.warn("Unique fault root cause has more than one occurrence probabilities in the FMECA data.")

            occurrence_porbability = np.max(unique_os)
            prob_val = get_occurrence_probability(o=occurrence_porbability)

            if bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle, 0) == "True":
                cpt[0] = prob_val
                cpt[1] = 1 - prob_val
            elif bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle, 0) == "True":
                cpt[1] = prob_val
                cpt[0] = 1 - prob_val
            else:
                logger.error("Wrong states of FC node!")

            bdsn_bayesian_network.getBayesianNetwork().set_node_definition(current_node_handle, cpt)
            #bdsn_bayesian_network.printNodeInfo(current_node_handle)

        elif re.match(r'^FE[\d][\d]_(?!FC[\d][\d]_)(.*)$', current_node_id):
            # Noisy Max Configuration
            dim_count = 2 + len(list_parent_node_handles)
            dim_sizes = [0] * dim_count

            list_parents_and_leak_id = list_parent_node_ids.copy()
            list_parents_and_leak_id.append("Leak")
            dict_parents_and_leak_states = {}
            dict_parents_and_leak_states["Leak"] = ["Leak"]

            # Leak dim
            dim_sizes[len(dim_sizes) - 2] = 1

            # current node number of states
            dim_sizes[len(dim_sizes) - 1] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(current_node_handle)

            for i in range(0, dim_count - 2):
                dim_sizes[i] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(list_parent_node_handles[i])
                parent_handle = list_parent_node_handles[i]
                parent_id = list_parent_node_ids[i]
                list_states = []
                for j in range(dim_sizes[i]):
                    list_states.append(bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(parent_handle, j))
                dict_parents_and_leak_states[parent_id] = list_states

            list_current_node_state_ids = []
            for i_state in range(0, bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(current_node_handle)):
                list_current_node_state_ids.append(bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle, i_state))

            i_cpt = -1
            for i_parent_and_leak_id in list_parents_and_leak_id:
                for i_parent_state_id in dict_parents_and_leak_states[i_parent_and_leak_id]:
                    for i_current_node_state in list_current_node_state_ids:
                        i_cpt += 1
                        outcome = i_current_node_state
                        #child_outcome = f"{i_parent_and_leak_id}:{i_parent_state_id}"
                        child_outcome = i_parent_state_id

                        if len(list_parent_node_ids) == 1:
                            #print(f"outcome: {outcome} ; child_outcomes: {child_outcome}")
                            if "OK" == child_outcome and outcome == "True":
                                cpt[i_cpt] = 0.0
                            elif "OK" == child_outcome and outcome == "False":
                                cpt[i_cpt] = 1.0
                            elif "UNK" == child_outcome and outcome == "True":
                                cpt[i_cpt] = 0.2
                            elif "UNK" == child_outcome and outcome == "False":
                                cpt[i_cpt] = 0.8
                            elif re.match(r'^FM[\d][\d]_(.*)$', child_outcome) and outcome == "True":
                                list_evoked_fault_effects = df_f_fm_fe[(df_f_fm_fe["Function"] == list_parent_node_ids[0]) & (df_f_fm_fe["Failure_Mode"] == child_outcome)]["evoked_fault_effects"].values[0]
                                if len(list_evoked_fault_effects) > 0 and current_node_id in list_evoked_fault_effects:
                                    df_associated_severities = df_fe_s[df_fe_s["Fault_Effect"].isin(list_evoked_fault_effects)]
                                    s_max = df_associated_severities["S"].max()
                                    cpt[i_cpt] = df_fe_s[df_fe_s["Fault_Effect"] == current_node_id]["S"].values[0] / s_max # S_FE,i / S_max
                                else:
                                    cpt[i_cpt] = 0
                            elif re.match(r'^FM[\d][\d]_(.*)$', child_outcome) and outcome == "False":
                                list_evoked_fault_effects = df_f_fm_fe[(df_f_fm_fe["Function"] == list_parent_node_ids[0]) & (df_f_fm_fe["Failure_Mode"] == child_outcome)]["evoked_fault_effects"].values[0]
                                if len(list_evoked_fault_effects) > 0 and current_node_id in list_evoked_fault_effects:
                                    df_associated_severities = df_fe_s[df_fe_s["Fault_Effect"].isin(list_evoked_fault_effects)]
                                    s_max = df_associated_severities["S"].max()
                                    cpt[i_cpt] = 1 - df_fe_s[df_fe_s["Fault_Effect"] == current_node_id]["S"].values[0] / s_max
                                else:
                                    cpt[i_cpt] = 1.0
                            elif "Leak" == child_outcome and outcome == "True":
                                cpt[i_cpt] = 0.0
                            elif "Leak" == child_outcome and outcome == "False":
                                cpt[i_cpt] = 1.0
                            else:
                                print("Why are we here?")

                        elif len(list_parent_node_ids) > 1:
                            # print(f"outcome: {outcome} ; child_outcomes: {child_outcome}")
                            if "OK" == child_outcome and outcome == "True":
                                cpt[i_cpt] = 0.0
                            elif "OK" == child_outcome and outcome == "False":
                                cpt[i_cpt] = 1.0
                            elif "UNK" == child_outcome and outcome == "True":
                                cpt[i_cpt] = 0.2
                            elif "UNK" == child_outcome and outcome == "False":
                                cpt[i_cpt] = 0.8
                            elif re.match(r'^FM[\d][\d]_(.*)$', child_outcome) and outcome == "True":
                                list_evoked_fault_effects = df_f_fm_fe[(df_f_fm_fe["Function"] == i_parent_and_leak_id) & (df_f_fm_fe["Failure_Mode"] == child_outcome)]["evoked_fault_effects"].values[0]
                                if len(list_evoked_fault_effects) > 0 and current_node_id in list_evoked_fault_effects:
                                    df_associated_severities = df_fe_s[df_fe_s["Fault_Effect"].isin(list_evoked_fault_effects)]
                                    s_max = df_associated_severities["S"].max()
                                    cpt[i_cpt] = df_fe_s[df_fe_s["Fault_Effect"] == current_node_id]["S"].values[0] / s_max  # S_FE,i / S_max
                                else:
                                    cpt[i_cpt] = 0
                            elif re.match(r'^FM[\d][\d]_(.*)$', child_outcome) and outcome == "False":
                                list_evoked_fault_effects = df_f_fm_fe[(df_f_fm_fe["Function"] == i_parent_and_leak_id) & (df_f_fm_fe["Failure_Mode"] == child_outcome)]["evoked_fault_effects"].values[0]
                                if len(list_evoked_fault_effects) > 0 and current_node_id in list_evoked_fault_effects:
                                    df_associated_severities = df_fe_s[df_fe_s["Fault_Effect"].isin(list_evoked_fault_effects)]
                                    s_max = df_associated_severities["S"].max()
                                    cpt[i_cpt] = 1 - df_fe_s[df_fe_s["Fault_Effect"] == current_node_id]["S"].values[0] / s_max
                                else:
                                    cpt[i_cpt] = 1.0
                            elif "Leak" == child_outcome and outcome == "True":
                                cpt[i_cpt] = 0.0
                            elif "Leak" == child_outcome and outcome == "False":
                                cpt[i_cpt] = 1.0
                            else:
                                print("Why are we here?")

            # print("#################")
            bdsn_bayesian_network.getBayesianNetwork().set_node_definition(current_node_handle, cpt)
            # bdsn_bayesian_network.printNodeInfo(bdsn_bayesian_network.getBayesianNetwork().get_node(current_node_id))
            # print(cpt)
            # print("=================")

        elif "CM" in current_node_id:
            dim_count = 1 + len(list_parent_node_handles)
            dim_sizes = [0] * dim_count

            for i in range(0, dim_count - 1):
                dim_sizes[i] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(list_parent_node_handles[i])

            dim_sizes[len(dim_sizes) - 1] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(
                current_node_handle)
            coords = [0] * dim_count

            for elem_idx in range(0, len(cpt)):
                indexToCoords(elem_idx, dim_sizes, coords)
                outcome = bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle,
                                                                                    coords[dim_count - 1])
                if dim_count > 1:
                    child_outcomes = []
                    for parent_idx in range(0, len(list_parent_node_handles)):
                        parent_handle = list_parent_node_handles[parent_idx]
                        child_outcomes.append(bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(parent_handle,
                                                                                                        coords[parent_idx]))

                if "True" in child_outcomes and outcome == "True":
                    cpt[elem_idx] = 0.95
                elif "True" in child_outcomes and outcome == "False":
                    cpt[elem_idx] = 0.05
                elif "True" not in child_outcomes and outcome == "True":
                    cpt[elem_idx] = 0.02
                elif "True" not in child_outcomes and outcome == "False":
                    cpt[elem_idx] = 0.98
                else:
                    print("Why are we here?")

            # print("#################")
            bdsn_bayesian_network.getBayesianNetwork().set_node_definition(current_node_handle, cpt)
            # bdsn_bayesian_network.printNodeInfo(bdsn_bayesian_network.getBayesianNetwork().get_node(current_node_id))
            # print(cpt)
            # print("=================")

        elif "SE" in current_node_id:
            dim_count = 1 + len(list_parent_node_handles)
            dim_sizes = [0] * dim_count

            for i in range(0, dim_count - 1):
                dim_sizes[i] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(list_parent_node_handles[i])

            dim_sizes[len(dim_sizes) - 1] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(current_node_handle)
            coords = [0] * dim_count

            for elem_idx in range(0, len(cpt)):
                indexToCoords(elem_idx, dim_sizes, coords)
                outcome = bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle, coords[dim_count - 1])
                if dim_count > 1:
                    child_outcomes = []
                    for parent_idx in range(0, len(list_parent_node_handles)):
                        parent_handle = list_parent_node_handles[parent_idx]
                        child_outcomes.append(bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(parent_handle, coords[parent_idx]))

                if "NOK" not in child_outcomes and outcome == "OK":
                    cpt[elem_idx] = 1.0
                elif "NOK" not in child_outcomes and outcome == "NOK":
                    cpt[elem_idx] = 0.0
                elif "NOK" in child_outcomes and outcome == "NOK":
                    cpt[elem_idx] = 1.0
                elif "NOK" in child_outcomes and outcome == "OK":
                    cpt[elem_idx] = 0.0
                else:
                    print("Why are we here?")

            # print("#################")
            bdsn_bayesian_network.getBayesianNetwork().set_node_definition(current_node_handle, cpt)
            # bdsn_bayesian_network.printNodeInfo(bdsn_bayesian_network.getBayesianNetwork().get_node(current_node_id))
            # print(cpt)
            # print("=================")

        elif "FU" in current_node_id:
            dim_count = 1 + len(list_parent_node_handles)
            dim_sizes = [0] * dim_count

            for i in range(0, dim_count - 1):
                dim_sizes[i] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(list_parent_node_handles[i])

            dim_sizes[len(dim_sizes) - 1] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(current_node_handle)
            coords = [0] * dim_count

            for elem_idx in range(0, len(cpt)):
                indexToCoords(elem_idx, dim_sizes, coords)
                outcome = bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle, coords[dim_count - 1])
                if dim_count > 1:
                    child_outcomes = []
                    for parent_idx in range(0, len(list_parent_node_handles)):
                        parent_handle = list_parent_node_handles[parent_idx]
                        child_outcomes.append(bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(parent_handle, coords[parent_idx]))

                if outcome == "OK":
                    cpt[elem_idx] = 1.0
                    for co in child_outcomes:
                        if co != "OK":
                            cpt[elem_idx] = 0.0

                elif outcome == "NOK":
                    cpt[elem_idx] = 0.0
                    for co in child_outcomes:
                        if co != "OK":
                            cpt[elem_idx] = 1.0

                else:
                    print("Why are we here?")

            # print("#################")
            bdsn_bayesian_network.getBayesianNetwork().set_node_definition(current_node_handle, cpt)
            # bdsn_bayesian_network.printNodeInfo(bdsn_bayesian_network.getBayesianNetwork().get_node(current_node_id))
            # print(cpt)
            # print("=================")

        elif re.match(r'^F[\d][\d]_(?!FC[\d][\d]_)(.*)$', current_node_id):
            # 2) CPTs of function nodes
            df_fc_f_fm_current = df_fc_f_o_fm[df_fc_f_o_fm["Function"] == current_node_id]

            if bdsn_bayesian_network.getBayesianNetwork().get_node_type(current_node_handle) == int(pysmile.NodeType.NOISY_MAX):
                # Noisy Max Configuration
                dim_count = 2 + len(list_parent_node_handles)
                dim_sizes = [0] * dim_count

                list_parents_and_leak_id = list_parent_node_ids.copy()
                list_parents_and_leak_id.append("Leak")
                dict_parents_and_leak_states = {}
                dict_parents_and_leak_states["Leak"] = ["Leak"]

                # Leak dim
                dim_sizes[len(dim_sizes) - 2] = 1

                # current node number of states
                dim_sizes[len(dim_sizes) - 1] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(
                    current_node_handle)

                for i in range(0, dim_count - 2):
                    dim_sizes[i] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(list_parent_node_handles[i])
                    parent_handle = list_parent_node_handles[i]
                    parent_id = list_parent_node_ids[i]
                    list_states = []
                    for j in range(dim_sizes[i]):
                        list_states.append(bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(parent_handle, j))
                    dict_parents_and_leak_states[parent_id] = list_states

                list_current_node_state_ids = []
                for i_state in range(0, bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(current_node_handle)):
                    list_current_node_state_ids.append(
                        bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle, i_state))

                i_cpt = -1
                for i_parent_and_leak_id in list_parents_and_leak_id:
                    for i_parent_state_id in dict_parents_and_leak_states[i_parent_and_leak_id]:
                        for i_current_node_state in list_current_node_state_ids:
                            i_cpt += 1
                            outcome = i_current_node_state
                            # child_outcome = f"{i_parent_and_leak_id}:{i_parent_state_id}"
                            child_outcome = i_parent_state_id

                            # Set probability values
                            if len(list_parent_node_ids) >= 1:
                                # print(f"outcome: {outcome} ; child_outcomes: {child_outcome}")
                                if child_outcome == "True" and outcome == "UNK":
                                    cpt[i_cpt] = 0.019
                                elif child_outcome == "True" and outcome == "OK":
                                    cpt[i_cpt] = 0.001
                                elif child_outcome == "False" and outcome == "OK":
                                    cpt[i_cpt] = 1.0
                                elif child_outcome == "False" and outcome != "OK":
                                    cpt[i_cpt] = 0.0
                                elif re.match(r'^FM[\d][\d]_(.*)$', outcome) and child_outcome == "True":
                                    if re.match(r'FC[\d][\d]_FE[\d][\d]_(.*)$', i_parent_and_leak_id):
                                        for cfrc in df_fc_f_fm_current["Fault_Root_Cause"].to_list():
                                            if i_parent_and_leak_id[0:5] in cfrc:
                                                i_parent_and_leak_id = cfrc
                                                break

                                    if i_parent_and_leak_id in df_fc_f_fm_current["Fault_Root_Cause"].to_list():
                                        list_caused_fms = df_fc_f_fm_current[df_fc_f_fm_current["Fault_Root_Cause"] == i_parent_and_leak_id]["caused_failure_modes"].values[0]
                                        if len(list_caused_fms) == 0:
                                            cpt[i_cpt] = 0.0
                                        elif len(list_caused_fms) == 1:
                                            if outcome in list_caused_fms:
                                                cpt[i_cpt] = 1.0 - 0.019 - 0.001
                                            else:
                                                cpt[i_cpt] = 0.0
                                        elif len(list_caused_fms) > 1:
                                            # calculate theta
                                            if outcome in list_caused_fms:
                                                dict_s_max = {}
                                                dict_o_probs = {}
                                                for fm in list_caused_fms:
                                                    dict_s_max[fm] = 0
                                                    dict_o_probs[fm] = 0

                                                    # handle severities
                                                    list_evoked_fault_effects = df_f_fm_fe[(df_f_fm_fe["Function"] == current_node_id) & (df_f_fm_fe["Failure_Mode"] == fm)]["evoked_fault_effects"].values[0]
                                                    if len(list_evoked_fault_effects) > 0:
                                                        df_associated_severities = df_fe_s[df_fe_s["Fault_Effect"].isin(list_evoked_fault_effects)]
                                                        s_max = df_associated_severities["S"].max()
                                                        dict_s_max[fm] = s_max

                                                    # handle occurrence probabilities
                                                    df_occurrence_probability = df_fc_f_fm_current[df_fc_f_fm_current["Fault_Root_Cause"] == i_parent_and_leak_id]
                                                    if fm in df_occurrence_probability["caused_failure_modes"].values[0]:
                                                        o = df_occurrence_probability["Occurrence_Probability_O"].values[0]
                                                        dict_o_probs[fm] = o

                                                # todo fix o issue

                                                theta = dict_s_max[outcome] * dict_o_probs[outcome] / np.sum([dict_s_max[i_fm] * dict_o_probs[i_fm] for i_fm in list_caused_fms]) * 0.98
                                                cpt[i_cpt] = theta
                                            else:
                                                cpt[i_cpt] = 0.0
                                    else:
                                        cpt[i_cpt] = 0
                                elif child_outcome == "Leak" and outcome != "OK":
                                    cpt[i_cpt] = 1*10**-7
                                elif child_outcome == "Leak" and outcome == "OK":
                                    cpt[i_cpt] = 1.0 - 1*10**-7 * (dim_sizes[-1] - 1)
                                else:
                                    print("Why are we here?")
            else:
                print("Why?")

            # print("#################")
            bdsn_bayesian_network.getBayesianNetwork().set_node_definition(current_node_handle, cpt)
            # bdsn_bayesian_network.printNodeInfo(bdsn_bayesian_network.getBayesianNetwork().get_node(current_node_id))
            # print(cpt)
            # print("=================")

        elif re.match(r'^FC[\d][\d]_FE[\d][\d]_(.*)$', current_node_id):
            # Causal chain node
            related_fc = df_fe_fc_joined[df_fe_fc_joined["causal_chain"] == current_node_id]["Fault_Root_Cause"].values[0]
            related_fe = df_fe_fc_joined[df_fe_fc_joined["causal_chain"] == current_node_id]["Fault_Effect"].values[0]

            dim_count = 1 + len(list_parent_node_handles)
            dim_sizes = [0] * dim_count

            for i in range(0, dim_count - 1):
                dim_sizes[i] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(list_parent_node_handles[i])

            dim_sizes[len(dim_sizes) - 1] = bdsn_bayesian_network.getBayesianNetwork().get_outcome_count(
                current_node_handle)
            coords = [0] * dim_count

            for elem_idx in range(0, len(cpt)):
                indexToCoords(elem_idx, dim_sizes, coords)
                outcome = bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(current_node_handle,
                                                                                    coords[dim_count - 1])
                if dim_count > 1:
                    child_outcomes = []
                    for parent_idx in range(0, len(list_parent_node_handles)):
                        parent_handle = list_parent_node_handles[parent_idx]
                        child_outcomes.append(bdsn_bayesian_network.getBayesianNetwork().get_outcome_id(parent_handle,
                                                                                                        coords[parent_idx]))

                if "OK" in child_outcomes and outcome == "True":
                    fc_os = df_fmeca_data[df_fmeca_data["Fault_Root_Cause"] == related_fc][["Fault_Root_Cause", "Occurrence_Probability_O"]]
                    unique_os = fc_os["Occurrence_Probability_O"].unique()
                    if len(unique_os) > 1:
                        logger.warn("Unique fault root cause has more than one occurrence probabilities in the FMECA data.")

                    occurrence_porbability = np.max(unique_os)
                    cpt[elem_idx] = get_occurrence_probability(occurrence_porbability)

                if "OK" in child_outcomes and outcome == "False":
                    #related_fc = df_fe_fc_joined[df_fe_fc_joined["causal_chain"] == current_node_id]["Fault_Root_Cause"].values[0]
                    fc_os = df_fmeca_data[df_fmeca_data["Fault_Root_Cause"] == related_fc][["Fault_Root_Cause", "Occurrence_Probability_O"]]
                    unique_os = fc_os["Occurrence_Probability_O"].unique()
                    if len(unique_os) > 1:
                        logger.warn("Unique fault root cause has more than one occurrence probabilities in the FMECA data.")

                    occurrence_porbability = np.max(unique_os)
                    cpt[elem_idx] = 1 - get_occurrence_probability(occurrence_porbability)

                elif "UNK" in child_outcomes and outcome == "True":
                    cpt[elem_idx] = 0.2
                elif "UNK" in child_outcomes and outcome == "False":
                    cpt[elem_idx] = 0.8
                elif re.match(r'^FM[\d][\d]_(.*)$', child_outcomes[0]) and outcome == "True":
                    list_evoked_fault_effects = df_f_fm_fe[(df_f_fm_fe["Function"] == list_parent_node_ids[0]) & (df_f_fm_fe["Failure_Mode"] == child_outcomes[0])]["evoked_fault_effects"].values[0]
                    if len(list_evoked_fault_effects) > 0 and related_fe in list_evoked_fault_effects:
                        df_associated_severities = df_fe_s[df_fe_s["Fault_Effect"].isin(list_evoked_fault_effects)]
                        s_max = df_associated_severities["S"].max()
                        cpt[elem_idx] = df_fe_s[df_fe_s["Fault_Effect"] == related_fe]["S"].values[0] / s_max  # S_FE,i / S_max
                    else:
                        cpt[elem_idx] = 0
                elif re.match(r'^FM[\d][\d]_(.*)$', child_outcomes[0]) and outcome == "False":
                    list_evoked_fault_effects = df_f_fm_fe[(df_f_fm_fe["Function"] == list_parent_node_ids[0]) & (df_f_fm_fe["Failure_Mode"] == child_outcomes[0])]["evoked_fault_effects"].values[0]
                    if len(list_evoked_fault_effects) > 0 and related_fe in list_evoked_fault_effects:
                        df_associated_severities = df_fe_s[df_fe_s["Fault_Effect"].isin(list_evoked_fault_effects)]
                        s_max = df_associated_severities["S"].max()
                        cpt[elem_idx] = 1 - df_fe_s[df_fe_s["Fault_Effect"] == related_fe]["S"].values[0] / s_max
                    else:
                        cpt[elem_idx] = 1.0
                else:
                    # Possible Leak configuration
                    continue

            # print("#################")
            bdsn_bayesian_network.getBayesianNetwork().set_node_definition(current_node_handle, cpt)
            # bdsn_bayesian_network.printNodeInfo(bdsn_bayesian_network.getBayesianNetwork().get_node(current_node_id))
            # print(cpt)
            # print("=================")

        else:
            pass

        h = bdsn_bayesian_network.getBayesianNetwork().get_next_node(h)

    t_end_heuristic_parameter_identification = perf_counter_ns() * 10 ** -6
    delta_t_heuristic_parameter_identification = t_end_heuristic_parameter_identification - t_start_heuristic_parameter_identification

    # write Bayesian network to file
    bdsn_bayesian_network.saveBayesianNetworkToFile("2025-11-08_bdsn_bayesian_network_based_on_fmeca_data.xdsl")

    return {
        "delta_t_read_in_fmeca_data": delta_t_read_in_fmeca_data,
        "delta_t_creation_bn_structure": delta_t_creation_bn_structure,
        "delta_t_heuristic_parameter_identification": delta_t_heuristic_parameter_identification
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heuristic parameter identification using FMECA data")
    parser.add_argument("path", type=str, help="str file path")
    parser.add_argument("-v", "--verbose", type=int, help="verbose mode integer number")
    parser.add_argument("-l", "--log_level", type=str, help="str to define the log level [info, debug, error, warning, critical]", default='error')
    parser.add_argument("-p", "--performance_evaluation", action='store_true')
    parser.add_argument("-n", "--n_iter", default=100, help="Number of iterations to perform the runtime measurement during performance evaluation")

    args = parser.parse_args()
    path = args.path
    verbose_mode = args.verbose if args.verbose else 0
    log_level = args.log_level
    do_performance_evaluation = args.performance_evaluation
    n_iter = int(args.n_iter)

    dict_log_levels = {
        "debug": logging.DEBUG,
        "error": logging.ERROR,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "critical": logging.CRITICAL
    }

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=dict_log_levels[log_level],
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.FileHandler("../../logs/log.log"), logging.StreamHandler()]
                        )

    logger.info('Starting heuristic parameter identification for a Bayesian network based on FMECA data ...')
    logger.debug(f"file path: {path}")
    logger.debug(f"verbose mode: {verbose_mode}")
    logger.debug(f"log_level: {log_level}")

    if do_performance_evaluation:
        dict_performance_measurements = {}
        list_delta_t_creation_bn_structure = []
        list_delta_t_heuristic_parameter_identification = []

        for i in range(n_iter):
            tmp_performance_result = heuristic_parameter_identification(path=path, logger=logger, verbose=verbose_mode)
            dict_performance_measurements[f"iteration_{i}"] = tmp_performance_result
            list_delta_t_heuristic_parameter_identification.append(tmp_performance_result["delta_t_heuristic_parameter_identification"])
            list_delta_t_creation_bn_structure.append(tmp_performance_result["delta_t_creation_bn_structure"])

        # BN creation time
        mean_bn_creation_time = np.mean(list_delta_t_creation_bn_structure)
        std_dev_bn_creation_time = np.std(list_delta_t_creation_bn_structure)
        lower_percentile_bn_creation_time = np.percentile(list_delta_t_creation_bn_structure, 2.5)
        upper_percentile_bn_creation_time = np.percentile(list_delta_t_creation_bn_structure, 97.5)

        data = (np.array(list_delta_t_creation_bn_structure),)
        ci_result = bootstrap(
            data,
            np.mean,
            confidence_level=0.95,
            n_resamples=10000,
            method='percentile',
            random_state=42
        )
        ci_low_bn_creation_time, ci_high_bn_creation_time = ci_result.confidence_interval

        dict_result_bn_creation_time = {
            "mean_bn_creation_time": mean_bn_creation_time,
            "std_dev_bn_creation_time": std_dev_bn_creation_time,
            "lower_percentile_bn_creation_time": lower_percentile_bn_creation_time,
            "upper_percentile_bn_creation_time": upper_percentile_bn_creation_time,
            "ci_low_bn_creation_time": ci_low_bn_creation_time,
            "ci_high_bn_creation_time": ci_high_bn_creation_time
        }

        # heuristic parameter identification time
        mean_heuristic_parameter_identification_time = np.mean(list_delta_t_heuristic_parameter_identification)
        std_dev_heuristic_parameter_identification_time = np.std(list_delta_t_heuristic_parameter_identification)
        lower_percentile_heuristic_parameter_identification_time = np.percentile(list_delta_t_heuristic_parameter_identification, 2.5)
        upper_percentile_heuristic_parameter_identification_time = np.percentile(list_delta_t_heuristic_parameter_identification, 97.5)

        data = (np.array(list_delta_t_heuristic_parameter_identification),)
        ci_result = bootstrap(
            data,
            np.mean,
            confidence_level=0.95,
            n_resamples=10000,
            method='percentile',
            random_state=42
        )
        ci_low_heuristic_parameter_identification_time, ci_high_heuristic_parameter_identification_time = ci_result.confidence_interval

        dict_result_heuristic_parameter_identification_time = {
            "mean_heuristic_parameter_identification_time": mean_heuristic_parameter_identification_time,
            "std_dev_heuristic_parameter_identification_time": std_dev_heuristic_parameter_identification_time,
            "lower_percentile_heuristic_parameter_identification_time": lower_percentile_heuristic_parameter_identification_time,
            "upper_percentile_heuristic_parameter_identification_time": upper_percentile_heuristic_parameter_identification_time,
            "ci_low_heuristic_parameter_identification_time": ci_low_heuristic_parameter_identification_time,
            "ci_high_heuristic_parameter_identification_time": ci_high_heuristic_parameter_identification_time
        }


        # print results
        print(dict_result_bn_creation_time)
        print(dict_result_heuristic_parameter_identification_time)

    else:
        heuristic_parameter_identification(path=path, logger=logger, verbose=verbose_mode)