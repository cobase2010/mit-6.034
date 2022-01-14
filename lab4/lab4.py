from classify import *
import math

##
## CSP portion of lab 4.
##
from csp import BinaryConstraint, CSP, CSPState, Variable,\
    basic_constraint_checker, solve_csp_problem

def check_and_reduce(state, X, x):
    neighbor_constraints = state.get_constraints_by_name(X.get_name())
    for constraint in neighbor_constraints:
        Y = state.get_variable_by_name(constraint.get_variable_j_name())
        if Y.is_assigned():
            continue
        # y_domain = Y.copy()
        for y in Y.get_domain():
            if not constraint.check(state, x, y):
                Y.reduce_domain(y)
            if Y.domain_size() == 0:
                return False
    return True

# Implement basic forward checking on the CSPState see csp.py
def forward_checking(state, verbose=False):
    # Before running Forward checking we must ensure
    # that constraints are okay for this state.
    basic = basic_constraint_checker(state, verbose)
    if not basic:
        return False

    # Add your forward checking logic here.
    X = state.get_current_variable()
    x = None
    if X is not None:  # We are not in the root state
        x = X.get_assigned_value()
        return check_and_reduce(state, X, x)
    else:
        return True

# Now Implement forward checking + (constraint) propagation through
# singleton domains.
def forward_checking_prop_singleton(state, verbose=False):
    # Run forward checking first.
    fc_checker = forward_checking(state, verbose)
    if not fc_checker:
        return False

    # Add your propagate singleton logic here.
    singleton_vars = [var for var in state.get_all_variables() if not var.is_assigned() and var.domain_size() == 1]
    visisted_singletons = []
    while len(singleton_vars) > 0:
        X = singleton_vars.pop()  # FIFO
        visisted_singletons.append(X)
        if not check_and_reduce(state, X, X.get_domain()[0]):
            return False
        new_singletons = [var for var in state.get_all_variables() if not var.is_assigned() and var not in visisted_singletons 
                    and var not in singleton_vars and var.domain_size() ==1]
        singleton_vars = new_singletons + singleton_vars
    return True

# BONUS: Now Implement forward checking with prop thru reduced domains
def forward_checking_with_prop_thru_reduced_domains(state, verbose=False):
    return arc_consistency(state)

def arc_consistency(state):
    queue = []
    all_constraints = state.get_all_constraints()
    for constraint in all_constraints:
        X = state.get_variable_by_name(constraint.get_variable_i_name())
        Y = state.get_variable_by_name(constraint.get_variable_j_name())
        if (X, Y) not in queue:
            queue.append((X, Y))
        # queue.append((Y, X))
    while len(queue) > 0:
        X, Y = queue.pop(0)
        if remove_inconsistent_values(state, all_constraints, X, Y):
            if X.domain_size() == 0:
                return False        # No solution found!
            neighbor_constraints = state.get_constraints_by_name(X.get_name(), order=2)
            for constraint in neighbor_constraints:
                Y2 = state.get_variable_by_name(constraint.get_variable_i_name())
                if Y2.get_name() != Y.get_name() and not Y2.is_assigned() and (Y2, X) not in queue:
                    queue = queue + [(Y2, X)] 

    return True



def remove_inconsistent_values(state, all_constraints, X, Y):
    reduced = False
    # domain_copy = [var for var in X.get_domain()]
    constraints = [var for var in all_constraints if var.get_variable_i_name() == X.get_name() \
                    and var.get_variable_j_name() == Y.get_name()]
    for x in X.get_domain():
        satisfy = False
        for y in Y.get_domain():
            for constraint in constraints:
                if constraint.check(state, x, y):
                    satisfy = True
                    break
            if satisfy:
                break
        if not satisfy:
            X.reduce_domain(x)
            reduced = True
    return reduced

        


## The code here are for the tester
## Do not change.
from moose_csp import moose_csp_problem
from map_coloring_csp import map_coloring_csp_problem

def csp_solver_tree(problem, checker):
    problem_func = globals()[problem]
    checker_func = globals()[checker]
    answer, search_tree = problem_func().solve(checker_func)
    return search_tree.tree_to_string(search_tree)

##
## CODE for the learning portion of lab 4.
##

### Data sets for the lab
## You will be classifying data from these sets.
senate_people = read_congress_data('S110.ord')
senate_votes = read_vote_data('S110desc.csv')

house_people = read_congress_data('H110.ord')
house_votes = read_vote_data('H110desc.csv')

last_senate_people = read_congress_data('S109.ord')
last_senate_votes = read_vote_data('S109desc.csv')


### Part 1: Nearest Neighbors
## An example of evaluating a nearest-neighbors classifier.
senate_group1, senate_group2 = crosscheck_groups(senate_people)
# evaluate(nearest_neighbors(hamming_distance, 1), senate_group1, senate_group2, verbose=1)

## Write the euclidean_distance function.
## This function should take two lists of integers and
## find the Euclidean distance between them.
## See 'hamming_distance()' in classify.py for an example that
## computes Hamming distances.

def euclidean_distance(list1, list2):

    """ Calculate the Euclidean distance between two lists """
    # Make sure we're working with lists
    # Sorry, no other iterables are permitted
    assert isinstance(list1, list)
    assert isinstance(list2, list)

    dist = 0

    # 'zip' is a Python builtin, documented at
    # <http://www.python.org/doc/lib/built-in-funcs.html>
    for item1, item2 in zip(list1, list2):
       dist += (item1 - item2) ** 2
    return dist ** 0.5
    

#Once you have implemented euclidean_distance, you can check the results:
# evaluate(nearest_neighbors(euclidean_distance, 1), senate_group1, senate_group2, verbose=1)

## By changing the parameters you used, you can get a classifier factory that
## deals better with independents. Make a classifier that makes at most 3
## errors on the Senate.

my_classifier = nearest_neighbors(euclidean_distance, 5)
# evaluate(my_classifier, senate_group1, senate_group2, verbose=1)

### Part 2: ID Trees
# print CongressIDTree(senate_people, senate_votes, homogeneous_disorder)

## Now write an information_disorder function to replace homogeneous_disorder,
## which should lead to simpler trees.

def information_disorder(yes, no):
    
    n_y = len(yes)
    n_n = len(no)
    n_t = n_y + n_n
    y_score = 0.0
    n_score = 0.0

    for member in set(yes):
        member_count = yes.count(member)
        y_score += -(member_count * 1.0/n_y) * math.log(member_count * 1.0/n_y, 2)
    
    for member in set(no):
        member_count = no.count(member)
        n_score += -(member_count * 1.0/n_n) * math.log(member_count * 1.0/n_n, 2)
    
    return n_y * 1.0 / n_t * y_score + n_n * 1.0/n_t * n_score


# print CongressIDTree(senate_people, senate_votes, information_disorder)
# evaluate(idtree_maker(senate_votes, homogeneous_disorder), senate_group1, senate_group2, verbose=2)

## Now try it on the House of Representatives. However, do it over a data set
## that only includes the most recent n votes, to show that it is possible to
## classify politicians without ludicrous amounts of information.

def limited_house_classifier(house_people, house_votes, n, verbose = False):
    house_limited, house_limited_votes = limit_votes(house_people,
    house_votes, n)
    house_limited_group1, house_limited_group2 = crosscheck_groups(house_limited)

    if verbose:
        print "ID tree for first group:"
        print CongressIDTree(house_limited_group1, house_limited_votes,
                             information_disorder)
        print
        print "ID tree for second group:"
        print CongressIDTree(house_limited_group2, house_limited_votes,
                             information_disorder)
        print
        
    return evaluate(idtree_maker(house_limited_votes, information_disorder),
                    house_limited_group1, house_limited_group2)

                                   
## Find a value of n that classifies at least 430 representatives correctly.
## Hint: It's not 10.
N_1 = 44
rep_classified = limited_house_classifier(house_people, house_votes, N_1, verbose=False)

## Find a value of n that classifies at least 90 senators correctly.
N_2 = 67
senator_classified = limited_house_classifier(senate_people, senate_votes, N_2)

## Now, find a value of n that classifies at least 95 of last year's senators correctly.
N_3 = 23
old_senator_classified = limited_house_classifier(last_senate_people, last_senate_votes, N_3)


## The standard survey questions.
HOW_MANY_HOURS_THIS_PSET_TOOK = "10"
WHAT_I_FOUND_INTERESTING = "All of them"
WHAT_I_FOUND_BORING = "None"


## This function is used by the tester, please don't modify it!
def eval_test(eval_fn, group1, group2, verbose = 0):
    """ Find eval_fn in globals(), then execute evaluate() on it """
    # Only allow known-safe eval_fn's
    if eval_fn in [ 'my_classifier' ]:
        return evaluate(globals()[eval_fn], group1, group2, verbose)
    else:
        raise Exception, "Error: Tester tried to use an invalid evaluation function: '%s'" % eval_fn

    
