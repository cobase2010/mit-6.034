from production import AND, OR, NOT, PASS, FAIL, IF, THEN, \
     match, populate, simplify, variables, RuleExpression
from zookeeper import ZOOKEEPER_RULES

# This function, which you need to write, takes in a hypothesis
# that can be determined using a set of rules, and outputs a goal
# tree of which statements it would need to test to prove that
# hypothesis. Refer to the problem set (section 2) for more
# detailed specifications and examples.

# Note that this function is supposed to be a general
# backchainer.  You should not hard-code anything that is
# specific to a particular rule set.  The backchainer will be
# tested on things other than ZOOKEEPER_RULES.

# General Backchaining Pseudo code:
# function rule_match_goal_tree(hypothesis, rules, DB)
# 1. check hypothesis against DB exit if satisfied
# 2. Find all matching rules: any rule with a consequent that matches hypothesis
# 3. For each rule in matching rules:
#  i) binding <- unify rule.consequent and hypothesis
#  ii) subtree <- antecedents_goal tree(rule, rules, binding, DB)
#  iii) Optimization: If subtree evaluation returns true,
#  we can short-circuit because we are ORing subtrees.
# return OR(rule subtrees)
# function antecedent_goal_tree(rule, rules, binding, DB)
# for each antecedent:
# 1. new-hypothesis <- antecedent + binding
# 2. check new-hypothesis against DB, if matched, update binding
# 3. subtree <- rule_match_goal_tree(new-hypothesis, rules, DB)
# 4. Optimization: Short circuit if the antecedent logics calls for it
#  i.e. if in an AND then the first failure fails the whole branch.
#  if in an OR, the first success implies the whole branch succeeds
#  return {antecedent logic}(antecedent subtrees)
# Note: If during antecedent_goal_tree step 2, there are multiple matches of the hypothesis in
# the DB then we can opt to create an OR subtree to represent all those database instantiations

def rule_match_goal_tree(hypothesis, rules):
    # print "testing hypothesis", hypothesis
    
    # if there is match, the goal tree will be in the form of OR (self, other_part)
    rule_subtrees = []
    for rule in rules:
        # here we assume consequent only contains simple leaf node, need more logic to handle complex expressions
        binding = match(rule.consequent()[0], hypothesis)
        if binding != None:
            sub_tree = antecedent_goal_tree(rule, rules, binding)
            rule_subtrees.append(sub_tree)
    if len(rule_subtrees) > 0:
        return OR([hypothesis] + rule_subtrees)    #always including self as part of the rule tree
    else:
        return hypothesis # leaf


def antecedent_goal_tree(rule, rules, binding):
    # print "testing rule", rule, "with binding", binding
    antecedent = rule.antecedent()
    hypothesis = populate(antecedent, binding)
    if not isinstance(hypothesis, RuleExpression): 
        return rule_match_goal_tree(hypothesis, rules) 
    else:
        subtrees = []
        for part in hypothesis:
            subtree = rule_match_goal_tree(part, rules)
            subtrees.append(subtree)  
        if isinstance(hypothesis, AND):
            return AND(subtrees)
        else:
            return OR(subtrees)


def backchain_to_goal_tree(rules, hypothesis):
    return simplify(rule_match_goal_tree(hypothesis, rules))




# Here's an example of running the backward chainer - uncomment
# it to see it work:
# print backchain_to_goal_tree(ZOOKEEPER_RULES, 'opus is a penguin')
