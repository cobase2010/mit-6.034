import Orange
from Orange.classification import svm
from Orange.evaluation import testing, scoring

bayes = Orange.classification.bayes.NaiveLearner(name="bayes")
simple_tree = Orange.classification.tree.SimpleTreeLearner(name="simple_tree")
lin = Orange.classification.svm.LinearLearner(name="lr")
knn = Orange.classification.knn.kNNLearner(name="knn")
svm_leaner = svm.SVMLearner(name="svm")
tree = Orange.classification.tree.TreeLearner(m_pruning=2, name="tree")
bs = Orange.ensemble.boosting.BoostedLearner(svm_leaner, t=10, name="boosted tree")
bg = Orange.ensemble.bagging.BaggedLearner(svm_leaner, name="bagged tree")
base_learners = [bayes, knn, svm_leaner, tree]
stack = Orange.ensemble.stacking.StackedClassificationLearner(base_learners)

lymphography = Orange.data.Table("breast-cancer.tab")

svm_leaner.tune_parameters(lymphography, parameters=["gamma"], folds=10)

learners = [svm_leaner, tree, bs, bg, bayes, simple_tree, lin, knn, stack]
results = Orange.evaluation.testing.cross_validation(learners, lymphography, folds=10)
print "Classification Accuracy:"
for i in range(len(learners)):
    print ("%15s: %5.3f") % (learners[i].name, Orange.evaluation.scoring.CA(results)[i])
    # print ("%15s: %5.3f") % (learners[i].name, Orange.evaluation.scoring.CA(results)[i])