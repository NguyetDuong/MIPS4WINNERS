
import sys
import functools


###### Boilerplate code used from Instance Validator ######

def main(argv):
    if len(argv) != 1:
        print "Usage: python algorithm.py [path_to_input_file]"
        return
    else:
        N, adj_matrix =  processInput(argv[0]) ## Processed the input of the files, returns number of nodes (N) and matrix (adj_matrix)
        graph = adj_matrix_to_graph(N, adj_matrix) ## Created an actual graph using the input
        # s_n = get_nodes_without_predecessors(graph)
        # for i in s_n:
        #     print(i.label)
        final_graph = minimum_acyclic_subgraph(graph)
        # print("...")
        # s_n = get_nodes_without_predecessors(final_graph)
        # for i in s_n:
        #     print(i.label)
        output = topological_sort(final_graph)
        print(output)

        #output = topological_sort(final_graph) #CURRENTLY IN REVISION

        #depth_first_search(graph, [graph.nodes[0]], 1) # this is for graphs without source
        # depth_first_search(graph, get_nodes_without_predecessors(graph), 1)

        # prepost = []
        # for n in graph.nodes:
        #     prepost.append((n.label, n.previsit, n.postvisit))
        # print(prepost)
        
        #c = sorted(prepost, key=lambda tup: -tup[2]) ## Sorting algorithm. 
        

        
        
        ##make sure we have everything in the right order. 
        ## " space-separated list of vertex numbers by increasing rank."
        ## probably print this so we can append the output from here to a 
        ## file using bash >>


def processInput(s):
    """Running with an input file, this method will check to see if the input file has
       the correct format, and if it satisfies all the conditions, it will return a tuple.
       This tuple will contains the number of nodes in the graph, and will return us the
       matrix representation of the graph."""


    fin = open(s, "r")
    line = fin.readline().split()
    if len(line) != 1 or not line[0].isdigit():
        return "Line 1 must contain a single integer."
    N = int(line[0])
    if N < 1 or N > 100:
        return "N must be an integer between 1 and 100, inclusive."

    # past here need to change #
    d = [[0 for j in range(N)] for i in range(N)]
    for i in xrange(N):
        line = fin.readline().split()
        if len(line) != N:
            return "Line " + str(i+2) + " must contain N integers."
        for j in xrange(N):
            if not line[j].isdigit():
                return "Line " + str(i+2) + " must contain N integers."
            d[i][j] = int(line[j])
            if d[i][j] < 0 or d[i][j] > 1:
                return "The adjacency matrix must be comprised of 0s and 1s."
    for i in xrange(N):
        if d[i][i] != 0:
            return "A node cannot have an edge to itself."
    return (N, d)


def adj_matrix_to_graph(N, am):
    """Creates a graph using the number of nodes, and the adjacency matrix input.
       Will return to us the graph object."""

    graph = Graph()
    for i in range(0, N):
        a = Node(i + 1, [], graph)
        for j in range(0, N):
            if am[i][j] == 1:
                a.add_successor_by_label(j + 1) ## Added j + 1 because we need labels to be 1, 2, 3, ..., N
        graph.add_node(a)

    return graph

def minimum_acyclic_subgraph(graph):
    ##Until there are no more cycles
        ##Depth first search the graph from a dummy source node (created every
        ##iteration since more sources can appear as edges are removed) that
        ##connects to all source nodes in the given graph, counting the number of
        ##cycles that each edge participates in. We do this by going back
        ##through the stack when we find a back edge, and increasing the cycle_count
        ##of each edge in the cycle by 1.

        ##Stack is kept as a list of vertices, and cycle counts for each edge
        ##are kept in the node that the edge comes out of.

        ##After the DFS is complete, we remove the edge that participates in
        ##the greatest number of cycles. 

    
    entered_DFS = update_cycles_graph(graph, graph.nodes[0])
    get_max_cycle_edge(graph)

    while entered_DFS > 0:
        graph.reset_all_nodes()
        entered_DFS = update_cycles_graph(graph, graph.nodes[0])
        get_max_cycle_edge(graph)

    graph.reset_all_nodes()

    return graph

def update_cycles_graph(graph, start_node):
    """This marks up all the nodes and increment the counter for the cycle."""

    stack = []
    path_stack = []
    entered_DFS = 0
    stack.append(start_node)
    while len(stack) > 0:
    #     print(stack)
    #     print(path_stack)
    #     print(start_node.mark)
    #     print("---------W---------")
        node = stack.pop()
        # print(stack)
        # print(path_stack)
        # print("---------P---------")
        if node.mark:
            path_stack.pop()
            # print(stack)
            # print(path_stack)
            # print("------------IF---------")
        else:
            stack.append(node)
            path_stack.append(node)
            node.mark = True

            for n in node.successors:
                if n in stack:
                    increment_detected_cycle(graph, path_stack, n, node)
                    entered_DFS += 1
                elif not n.mark:
                    stack.append(n)
                    
    """For unconnected graphs."""
    no_val = []
    for n in graph.nodes:
        if not n.mark:
            no_val.append(n)
    if len(no_val) > 0:
        update_cycles_graph(graph, no_val[0])

    return entered_DFS

def increment_detected_cycle(graph, stack, repeated_node, current_node):
    """Whenever a cycle is detected, this function will increase the cycle count of all involved edges by 1.
    """

    """
    cycle_stack = []
    cycle_stack.append(node)
    while len(cycle_stack) > 0:
        n = cycle_stack.pop()
        if n == start_node:
            return
        else:
            for n_c in n.successors:
                if n_c.mark:
                    if (n.cycle_count).has_key(n_c):
                        n.cycle_count[n_c] += 1
                    else:
                        n.cycle_count[n_c] = 1
                    go_through_cycle(graph, n_c, start_node)
    """

    my_stack = stack[:]
    next = repeated_node
    cur = current_node
    assert my_stack.pop() == cur
    while (cur != repeated_node) and len(my_stack) > 0:
        if next.label not in cur.edge_cycles.keys():
            cur.edge_cycles[next.label] = 1
        else:
            cur.edge_cycles[next.label] += 1
        next = cur
        cur = my_stack.pop()


    if next.label not in cur.edge_cycles.keys():
        cur.edge_cycles[next.label] = 1
    else:
        cur.edge_cycles[next.label] += 1


def get_max_cycle_edge(graph):
    """Returns the edge that has the maximum cycle counter.
       Specifically a tuple (A,B) where A -> B, so we just need
       to remove A's successors that is B."""
    # print("MAX CYCLE EDGE")
    A = None
    B = None

    max_val = 0

    for n in graph.nodes:
        # print("node we are on: " + str(n.label))
        for label in n.successor_labels:
            # print("node's successor: " + str(label))
            # print("num of cycles:" + str(n.edge_cycles[label]))

            if n.edge_cycles[label] > max_val:
                max_val = n.edge_cycles[label]
                A = n
                B = graph.get_node_by_label(label)

    if type(A) == type(graph.nodes[0]):
        # print("is removed")
        # print(A.label)
        # print(B.label)
        A.remove_successor(B)
        # print(A.successor_labels)

    return None


def topological_sort(graph):
    """Returns the list of vertex labels in topological order. 

       This should be done on the revised graph already, such that there are no obvious cycles.
       If there are multiple source node, make a giant big node and connect them to the multiple
       source nodes."""


    ## Making the big node (if there is more than one source node)
    source_nodes = get_nodes_without_predecessors(graph)
    s_labels = []
    for i in source_nodes:
        s_labels.append(i.label)
    # print("The numbers of source_nodes:")
    # for i in source_nodes:
    #     print(i.label)
    
    if len(source_nodes) == 0:
        big_node = [graph.nodes[0]]
    elif len(source_nodes) == 1:
        big_node = source_nodes
    else:
        big_node_obj = Node(101, s_labels, graph)
        graph.add_node(big_node_obj)
        big_node = [big_node_obj]
        # MIMI HAS ANAL PROBLEMS. 


    
    depth_first_search(graph, big_node, 1)
    prepost = []

    for n in graph.nodes:
        prepost.append((n.label, n.previsit, n.postvisit))
    
    # for n in prepost:
    #     print n

    c = sorted(prepost, key=lambda tup: -tup[2]) # sorting our shit by highest postvalue to lowest postvalue
    ordering = ""
    for node in c:
        ordering += str(node[0]) + " "
    return ordering


def depth_first_search(graph, source_nodes, val):
    """A slightly revised DFS algorithm.
       This algorithm will only work if there are source nodes!!"""

    
    ## This is the actual DFS algorithm
    stack = []
    #source_nodes = [graph.nodes[0]] #get_nodes_without_predecessors(graph)
    while len(source_nodes) != 0:
        stack.append(source_nodes.pop())
        while len(stack) > 0:
            node = stack.pop()
            if node.previsit == -1:
                stack.append(node)
                node.previsit = val
                val += 1
                for n in node.successors:
                    if n.postvisit == -1 and n.previsit == -1:
                        stack.append(n)
            elif node.postvisit == -1:
                node.postvisit = val
                val += 1


    ## This checks for any nodes left over that has not been traversed bc it's dumb af
    no_val = []
    for n in graph.nodes:
        if n.postvisit == -1:
            no_val.append(n)
    if len(no_val) > 0:
        depth_first_search(graph, no_val, val)


def get_nodes_without_predecessors(graph):
    """Returns a list of nodes that has no predecessors.
       If there are none, then we return an empty list."""
    my_nodes = []

    for n in graph.nodes:
        if (len(graph.get_predecessors(n))) == 0:
            my_nodes.append(n)

    return my_nodes

@functools.total_ordering
class Node:

    def __init__(self, label, successors, graph):
        """Initialize this node with a label, a list of successor labels, and
        a graph that is a part of.
        
        Nodes must be part of a graph.
        """
        self.label = label
        #A list of successor labels, should beintegers. 
        self.successor_labels = successors
        self.edge_cycles = {}
        for label in successors:
            self.edge_cycles[label] = 0
        #The graph this node is part of
        self.graph = graph
        self.previsit = -1
        self.postvisit = -1
        self.mark = False
        

    def __eq__(self, other):
        return self.label == other.label

    def __lt__(self, other):
        return self.label < other.label

    def remove_successor_by_label(self, label):
        self.successor_labels.remove(label)
        self.edge_cycles.pop(label)

    def remove_successor(self, succ):
        self.successor_labels.remove(succ.label)
        self.edge_cycles.pop(succ.label)

    def add_successor(self, succ):
        self.successor_labels.append(succ.label)
        self.edge_cycles[succ.label] = 0

    def add_successor_by_label(self, succ_label):
        self.successor_labels.append(succ_label)
        self.edge_cycles[succ_label] = 0

    def reset(self):
        self.previsit = -1
        self.postvisit = -1
        for key in self.edge_cycles:
            self.edge_cycles[key] = 0
        self.mark = False
    
    def unmark(self):
        self.mark = False

    def undo_cycle_count(self):
        for succ in self.cycle_count.keys():
            self.cycle_count[succ] = 0

    ## Access the successors, predecessors, and predecessor_labels as if they
    ## instance variables. They will be computed on access.
    @property
    def successors(self):
        """Returns all the successors of this specific node."""
        return [self.graph.get_node_by_label(x) for x in
                self.successor_labels]

    @property
    def predecessors(self):
        """Returns all the predecessors of this specific node.

           NOTE: APPARENTLY THIS DOES NOT WORK -- JUST CALL DIRECTLY."""
        return self.graph.get_predecessors(self)

    @property
    def predecessor_labels(self):
        """Returns all the labels of the predecessors."""
        return self.graph.get_predecessor_labels(self)

class Graph(object):
    """Defines a graph object containing nodes that have edges to other nodes.
    Initially empty.

    Nodes are stored inside this graph, and operations on the graph can be
    done with the node itself, or the label (an integer) of the node. 

    Tests:

    >>> a = Graph()
    >>> for x in range(0, 5):
    ...     a.add_node(Node(x, [x + 1], a))
    >>> a.get_node_by_label(4).remove_successor_by_label(5)
    >>> a.get_node_by_label(4).add_successor_by_label(0)
    >>> num_1 = a.get_node_by_label(1)
    >>> for x in [0, 3, 4]:
    ...     num_1.add_successor_by_label(x)
    >>> for x in [2, 3]:
    ...     a.get_node_by_label(x).add_successor_by_label(0)
    >>> zero = a.get_node_by_label(0)
    >>> zero_pred = sorted(zero.predecessor_labels)
    >>> zero_pred == [1, 2, 3, 4]
    True
    >>> zero.successors[0] == a.get_node_by_label(1)
    True
    >>> sorted(num_1.successor_labels) == [0, 2, 3, 4]
    True
    >>> 1 in a.get_node_by_label(3).predecessor_labels
    True
    >>> num_1.remove_successor_by_label(3)
    >>> 1 in a.get_node_by_label(3).predecessor_labels
    False
    >>> sorted(num_1.successor_labels) == [0, 2, 4]
    True
    >>> a.del_node_by_label(0)
    >>> 0 not in a.get_node_by_label(1).successor_labels
    True
    >>> 0 not in a.get_node_by_label(4).successor_labels
    True
    """

    def __init__(self):
        self.nodes = []
    
    def reset_all_nodes(self):
        for node in self.nodes:
            node.reset()


    ## Returns the pair of nodes A, B that the edge A -> B is between.
    ## This edge participates in the maximum number of cycles. It can be
    ## removed from the graph by calling A.remove_successor(B)
    def nodes_with_highest_cycle_count_edge(self):
        """Returns the pair of nodes (A, B) that the edge A -> B is between.
           This edge participates in the maximum number of cycles.
           It can be removed from the graph by calling A.remove_successor(B).



           NOT BEING USED"""
        # max_cycle = 0
        # v1 = None
        # v2 = None
        # for node1 in self.nodes:
        #     for node2label in node1.successor_labels:
        #         if node1.edge_cycles[node2label] > max_cycle:
        #             v1 = node1
        #             v2 = self.get_node_by_label(node2label)
        #             max_cycle = node1.edge_cycles[node2label]

        start_node = self.nodes[0]
        stack = []
        stack.append(start_node)
        while len(stack) > 0:
            node = stack.pop()
            if node.mark:
                update_cycle(node, start_node)
            else:
                node.mark = True
                for n in node.successors:
                    stack.append(n)

        no_val = []
        for n in self.nodes:
            if not n.mark:
                no_val.append(n)
        if len(no_val) > 0:
            nodes_with_highest_cycle_count_edge()

        A = None
        B = None
        max_val = -1

        for n in self.nodes:
            for n_suc, cycle in n.cycle_count.iteritems():
                if cycle > max_val:
                    A = n
                    B = n_suc
                    max_val = cycle


        return None

    def add_node(self, node):
        self.nodes.append(node)

    def del_node_by_label(self, label):
        to_remove = self.get_node_by_label(label)
        predecessors = self.get_predecessors(to_remove)

        #First remove all edges to the node
        for pred in predecessors:
            pred.remove_successor(to_remove)

        #Then remove the node itself and its edges out.
        self.nodes.remove(to_remove)

    def del_node(self, node):
        self.del_node_by_label(node.label)

    #NODES MUST HAVE UNIQUE LABELS
    def get_node_by_label(self, label):
        result = [node for node in self.nodes if node.label == label]
        assert len(result) == 1, "There are multiple nodes with same label"
        return result[0]

    def get_predecessors(self, b):
        result = []
        for a in self.nodes:
            if b.label in a.successor_labels:
                result.append(a)
        return result

    def get_predecessor_labels(self, b):
        return [x.label for x in self.get_predecessors(b)]






if __name__ == '__main__':
    main(sys.argv[1:])
