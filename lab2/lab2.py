# Fall 2012 6.034 Lab 2: Search
#
# Your answers for the true and false questions will be in the following form.  
# Your answers will look like one of the two below:
#ANSWER1 = True
#ANSWER1 = False

# 1: True or false - Hill Climbing search is guaranteed to find a solution
#    if there is a solution
ANSWER1 = False

# 2: True or false - Best-first search will give an optimal search result
#    (shortest path length).
#    (If you don't know what we mean by best-first search, refer to
#     http://courses.csail.mit.edu/6.034f/ai3/ch4.pdf (page 13 of the pdf).)
ANSWER2 = False

# 3: True or false - Best-first search and hill climbing make use of
#    heuristic values of nodes.
ANSWER3 = True

# 4: True or false - A* uses an extended-nodes set.
ANSWER4 = True

# 5: True or false - Breadth first search is guaranteed to return a path
#    with the shortest number of nodes.
ANSWER5 = True

# 6: True or false - The regular branch and bound uses heuristic values
#    to speed up the search for an optimal path.
ANSWER6 = False

# Import the Graph data structure from 'search.py'
# Refer to search.py for documentation
from search import Graph
import math

## Optional Warm-up: BFS and DFS
# If you implement these, the offline tester will test them.
# If you don't, it won't.
# The online tester will not test them.
# function Search(graph, start, goal):
#  0. Initialize
#  agenda = [ [start] ]
#  extended_list = []
#  while agenda is not empty:
#  1. path = agenda.pop(0) # get first element from agenda & return it
#  2. if is-path-to-goal(path, goal)
#  return path
#  3. otherwise extend the current path if not already extended
#  for each connected node
#  make a new path (don't add paths with loops!)
#  4. add new paths from 3 to agenda and reorganize agenda
#  (algorithms differ here see table below)
#  fail!

def bfs(graph, start, goal):
    agenda = [[start]]
    extended_list = []

    while len(agenda) > 0:
        path = agenda[0]    # Get the first element from the beginning
        agenda = agenda[1:]
        current_node = path[-1]
        if current_node == goal:
            return path
        else:
            connected_nodes = graph.get_connected_nodes(current_node)
            for node in connected_nodes:
                if node not in extended_list:
                    extended_list.append(node)
                    agenda = agenda + [path + [node]]  # add to the tail of the agenda
    return [] # didn't find path
                    



## Once you have completed the breadth-first search,
## this part should be very simple to complete.
def dfs(graph, start, goal):
    agenda = [[start]]
    extended_list = []

    while len(agenda) > 0:
        path = agenda[0]    # Get the first element from the beginning
        agenda = agenda[1:]
        current_node = path[-1]
        if current_node == goal:
            return path
        else:
            connected_nodes = graph.get_connected_nodes(current_node)
            for node in reversed(connected_nodes):   # Add the paths in reversed order to the front to preserved nature tree order
                if node not in extended_list:
                    extended_list.append(node)
                    agenda = [path + [node]] + agenda # add to the front of the agenda
    return [] # didn't find path


## Now we're going to add some heuristics into the search.  
## Remember that hill-climbing is a modified version of depth-first search.
## Search direction should be towards lower heuristic values to the goal.
def hill_climbing(graph, start, goal):
    agenda = [[start]]

    while len(agenda) > 0:
        path = agenda[0]    # Get the first element from the beginning
        agenda = agenda[1:]
        current_node = path[-1]
        if current_node == goal:
            return path
        else:
            connected_nodes = graph.get_connected_nodes(current_node)
            agenda2 = []
            for node in connected_nodes:   
                if node not in path: # avoid loops since we are not using extended list
                    agenda2.append(path + [node])
            #only sort the newly added nodes based on heuristic distance between its terminal node and goal
            # for best search, we'd sort all the node in agenda instead of just newly added ones
            agenda2.sort(key=lambda p: graph.get_heuristic(p[-1], goal)) 
            agenda = agenda2 + agenda  # add the new list to the front
            
    return [] # didn't find path

## Now we're going to implement beam search, a variation on BFS
## that caps the amount of memory used to store paths.  Remember,
## we maintain only k candidate paths of length n in our agenda at any time.
## The k top candidates are to be determined using the 
## graph get_heuristic function, with lower values being better values.
def beam_search(graph, start, goal, beam_width):
    agenda = [[start]]
    agenda2 = [] # agenda for next level


    while len(agenda) > 0:
        
        path = agenda[0]    # Get the first element from the beginning
        agenda = agenda[1:]
        current_node = path[-1]
        if current_node == goal:
            return path
        else:
            connected_nodes = graph.get_connected_nodes(current_node)
            for node in connected_nodes:
                if node not in path:
                    agenda2.append(path + [node])  # add to the tail of the agenda
        if len(agenda) == 0: # agenda is empty, process next level
            agenda2.sort(key=lambda p: graph.get_heuristic(p[-1], goal))
            for i in range(beam_width):
                if i < len(agenda2):
                    agenda.append(agenda2[i])       #only add beam_width best in the agenda
            agenda2 = []

    return [] # didn't find path

## Now we're going to try optimal search.  The previous searches haven't
## used edge distances in the calculation.

## This function takes in a graph and a list of node names, and returns
## the sum of edge lengths along the path -- the total distance in the path.
def path_length(graph, node_names):
    sum = 0
    for i in range(len(node_names) - 1):
        sum += graph.get_edge(node_names[i], node_names[i+1]).length
    return sum



def branch_and_bound(graph, start, goal):
    agenda = [[start]]
    # count = 0
    while len(agenda) > 0:
        # count += 1
        path = agenda[0]    # Get the first element from the beginning
        agenda = agenda[1:]
        current_node = path[-1]
        if current_node == goal:
            # print "num iterations", count
            return path
        else:
            connected_nodes = graph.get_connected_nodes(current_node)
            for node in connected_nodes:
                if node not in path:
                    agenda.append(path + [node])  # add to the tail of the agenda
            # for branch and bound, just sort the whole agenda based on the actual distance
            agenda.sort(key=lambda p: path_length(graph, p))
    # print "num iterations", count
    return [] # didn't find path

def a_star(graph, start, goal):
    agenda = [[start]]
    extended_list = []
    # count = 0
    while len(agenda) > 0:
        # count += 1
        path = agenda[0]    # Get the first element from the beginning
        agenda = agenda[1:]
        current_node = path[-1]
        if current_node == goal:
            # print "num iterations", count
            return path
        else:
            connected_nodes = graph.get_connected_nodes(current_node)
            for node in connected_nodes:
                if node not in extended_list:
                    extended_list.append(node)
                    agenda.append(path + [node])  # add to the tail of the agenda
            # for branch and bound, just sort the whole agenda based on the actual distance
            agenda.sort(key=lambda p: graph.get_heuristic(p[-1], goal) + path_length(graph, p))
    # print "num iterations", count
    return [] # didn't find path

# astar without using extended list
def a_star2(graph, start, goal):
    agenda = [[start]]
    # count = 0
    while len(agenda) > 0:
        # count += 1
        path = agenda[0]    # Get the first element from the beginning
        agenda = agenda[1:]
        current_node = path[-1]
        if current_node == goal:
            # print "num iterations", count
            return path
        else:
            connected_nodes = graph.get_connected_nodes(current_node)
            for node in connected_nodes:
                if node not in path:
                    agenda.append(path + [node])  # add to the tail of the agenda
            # for branch and bound, just sort the whole agenda based on the actual distance
            agenda.sort(key=lambda p: graph.get_heuristic(p[-1], goal) + path_length(graph, p))
    # print "num iterations", count
    return [] # didn't find path

## It's useful to determine if a graph has a consistent and admissible
## heuristic.  You've seen graphs with heuristics that are
## admissible, but not consistent.  Have you seen any graphs that are
## consistent, but not admissible?

def is_admissible(graph, goal):
    res = True
    for node in graph.nodes:
        if graph.get_heuristic(node, goal) > path_length(graph, branch_and_bound(graph, node, goal)):
            res = False
            break
    return res


def is_consistent(graph, goal):
    res = True
    for edge in graph.edges:
        if edge.length < abs(graph.get_heuristic(edge.node1, goal) - graph.get_heuristic(edge.node2, goal)):
            res = False
            break
    return res

HOW_MANY_HOURS_THIS_PSET_TOOK = '4'
WHAT_I_FOUND_INTERESTING = 'Not sure'
WHAT_I_FOUND_BORING = 'Not sure'

# unit test
AGRAPH = Graph(nodes = ['S', 'A', 'B', 'C', 'G'],
               edgesdict = [{'NAME': 'eSA', 'LENGTH': 3, 'NODE1': 'S', 'NODE2': 'A'},
                            {'NAME': 'eSB', 'LENGTH': 1, 'NODE1': 'S', 'NODE2': 'B'},
                            {'NAME': 'eAB', 'LENGTH': 1, 'NODE1': 'A', 'NODE2': 'B'},
                            {'NAME': 'eAC', 'LENGTH': 1, 'NODE1': 'A', 'NODE2': 'C'},
                            {'NAME': 'eCG', 'LENGTH': 10, 'NODE1': 'C', 'NODE2': 'G'}],
               heuristic = {'G':{'S': 12,
                                 'A': 9,
                                 'B': 12,
                                 'C': 8,
                                 'G': 0}})
print "AStar with extended list: ", a_star(AGRAPH, 'S', 'G')
print "AStar without extended list: ", a_star2(AGRAPH, 'S', 'G')
print "Branch and bound: ", branch_and_bound(AGRAPH, 'S', 'G')
