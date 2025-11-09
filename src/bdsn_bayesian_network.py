import typing
import pysmile

class BdsnBayesianNetwork():
    FAILURE_CAUSE_NODE = 0
    FAILURE_EFFECT_NODE = 1
    FUNCTIONAL_UNIT_NODE = 2
    MEAUSRE_NODE = 3
    STRUCTURAL_COMPONENT_NODE = 4
    CAUSAL_CHAIN_NODE = 5
    FUNCTION_NODE = 6
    UNDEFINED_TYPE_NODE = 238424

    # Color definitions
    FAILURE_CAUSE_NODE_COLOR = 0x6480ff #ff8064
    FUNCTIONAL_UNIT_NODE_COLOR = 0x8d8d8d
    FUNCTION_NODE_COLOR = 0xffcc99
    FAILURE_EFFECT_NODE_COLOR = 0x40e3fb #fbe340
    MEASURE_NODE_COLOR = 0x7dff7d #7dff7d
    STRUCTURAL_COMPONENT_COLOR = 0xff97ca # 0xd2d2d2
    CAUSAL_CHAIN_NODE_COLOR = 0x40e3fb #fbe340
    UNDEFINED_NODE_COLOR = 0x0000ff #ff0000

    # Border color definitions
    FAILURE_CAUSE_NODE_BORDER_COLOR = 0x000000
    FUNCTIONAL_UNIT_NODE_BORDER_COLOR = 0x000000
    FUNCTION_NODE_BORDER_COLOR = 0x000000
    FAILURE_EFFECT_NODE_BORDER_COLOR = 0x000000
    MEASURE_NODE_BORDER_COLOR = 0x000000
    STRUCTURAL_COMPONENT_BORDER_COLOR = 0x000000
    CAUSAL_CHAIN_NODE_BORDER_COLOR = 0x6480ff #ff8064
    UNDEFINED_NODE_BORDER_COLOR = 0xff0000

    MAP_NODE_TYP_COLOR = {
        FAILURE_CAUSE_NODE: FAILURE_CAUSE_NODE_COLOR,
        FAILURE_EFFECT_NODE: FAILURE_EFFECT_NODE_COLOR,
        FUNCTION_NODE: FUNCTION_NODE_COLOR,
        FUNCTIONAL_UNIT_NODE: FUNCTIONAL_UNIT_NODE_COLOR,
        MEAUSRE_NODE: MEASURE_NODE_COLOR,
        STRUCTURAL_COMPONENT_NODE: STRUCTURAL_COMPONENT_COLOR,
        CAUSAL_CHAIN_NODE: CAUSAL_CHAIN_NODE_COLOR,
        UNDEFINED_TYPE_NODE: UNDEFINED_NODE_COLOR
    }

    MAP_NODE_TYP_BORDER_COLOR = {
        FAILURE_CAUSE_NODE: FAILURE_CAUSE_NODE_BORDER_COLOR,
        FAILURE_EFFECT_NODE: FAILURE_EFFECT_NODE_BORDER_COLOR,
        FUNCTION_NODE: FUNCTION_NODE_BORDER_COLOR,
        FUNCTIONAL_UNIT_NODE: FUNCTIONAL_UNIT_NODE_BORDER_COLOR,
        MEAUSRE_NODE: MEASURE_NODE_BORDER_COLOR,
        STRUCTURAL_COMPONENT_NODE: STRUCTURAL_COMPONENT_BORDER_COLOR,
        CAUSAL_CHAIN_NODE: CAUSAL_CHAIN_NODE_BORDER_COLOR,
        UNDEFINED_TYPE_NODE: UNDEFINED_NODE_BORDER_COLOR
    }

    def __init__(self, logger):
        self.logger = logger
        self.logger.info("Instantiating BDSN Bayesian Network ...")
        self.bayesian_network = pysmile.Network()
        logger.info("Created empty pysmile Bayesian network")


    def getBayesianNetwork(self):
        return self.bayesian_network

    def createCptNode(self,
                      id: str,
                      name: str,
                      bdsn_node_type: int = UNDEFINED_TYPE_NODE,
                      node_states: typing.List = [],
                      x_pos: int = None,
                      y_pos: int = None,
                      width: int = 85,
                      height: int = 85):
        handle = self.bayesian_network.add_node(node_type=pysmile.NodeType.NOISY_MAX if bdsn_node_type == BdsnBayesianNetwork.FUNCTION_NODE or bdsn_node_type == BdsnBayesianNetwork.FAILURE_EFFECT_NODE else pysmile.NodeType.CPT,
                                                node_id=id)
        self.bayesian_network.set_node_name(handle, name)
        self.bayesian_network.set_node_position(handle, x_pos, y_pos, width, height)
        initial_outcome_count = self.bayesian_network.get_outcome_count(handle)
        self.bayesian_network.set_node_bg_color(handle, BdsnBayesianNetwork.MAP_NODE_TYP_COLOR[bdsn_node_type])
        self.bayesian_network.set_node_border_color(handle, BdsnBayesianNetwork.MAP_NODE_TYP_BORDER_COLOR[bdsn_node_type])

        if bdsn_node_type == BdsnBayesianNetwork.CAUSAL_CHAIN_NODE:
            self.bayesian_network.set_node_border_width(handle, 3)

        for i in range(0, initial_outcome_count):
            self.bayesian_network.set_outcome_id(handle, i, node_states[i])

        for i in range(initial_outcome_count, len(node_states)):
            self.bayesian_network.add_outcome(handle, node_states[i])

        return handle


    def createDiscreteDeterministicNode(self,
                                        id: str,
                                        name: str,
                                        bdsn_node_type: int = UNDEFINED_TYPE_NODE,
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



    def initializeDiscreteFunctionalUnitNode(self, handle_node):
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



    def getListCptStates(self, node_handle):
        num_states = self.bayesian_network.get_outcome_count(node_handle)
        return [self.bayesian_network.get_outcome_id(node_handle, state) for state in range(0, num_states)]


    def indexToCoords(self, index, dim_sizes, coords):
        prod = 1
        for i in range(len(dim_sizes) - 1, -1, -1):
            coords[i] = int(index / prod) % dim_sizes[i]
            prod *= dim_sizes[i]