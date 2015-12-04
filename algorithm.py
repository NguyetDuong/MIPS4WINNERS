
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
        # final_graph = minimum_acyclic_subgraph(graph)
        # output = topological_sort(final_graph) 

        output = topological_sort(graph) #CURRENTLY IN REVISION
        #print(output)

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

    return None

def topological_sort(graph):
    """Returns the list of vertex labels in topological order. 

       This should be done on the revised graph already, such that there are no obvious cycles.
       If there are multiple source node, make a giant big node and connect them to the multiple
       source nodes."""


    ## Making the big node (if there is more than one source node)
    source_nodes = get_nodes_without_predecessors(graph)
    if len(source_nodes) == 0:
        big_node = [graph.nodes[0]]
    elif len(source_nodes) == 1:
        big_node = source_nodes
    else:
        big_node = Node(101, source_nodes, graph)
        graph.add_node(big_node)


    
    depth_first_search(graph, big_node, 1)
    prepost = []

    for n in graph.nodes:
        prepost.append((n.label, n.previsit, n.postvisit))
    
    for n in prepost:
        print n

    ## return prepost if you want to see all of the layout
    return None


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
        for key in edge_cycles:
            self.edge_cycles[key] = 0

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
           It can be removed from the graph by calling A.remove_successor(B)."""
        max_cycle = 0
        v1 = None
        v2 = None
        for node1 in self.nodes:
            for node2label in node2.successor_labels:
                if node1.edge_cycles[node2label] > max_cycle:
                    v1 = node1
                    v2 = self.get_node_by_label(node2label)
                    max_cycle = node1.edge_cycles[node2label]
        return (v1, v2) 

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
