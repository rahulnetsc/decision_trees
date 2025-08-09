from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV

X,y = make_moons(n_samples=10000, noise=0.4)
X_train,  X_test,y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)
tree_clf = DecisionTreeClassifier( random_state= 42)
tree_params = {'max_leaf_nodes': list(range(2,15,2)),
               'max_depth': list(range(3,7))}

grid_search = GridSearchCV(tree_clf,param_grid= tree_params)
grid_search.fit(X=X_train,y=y_train)
opt_tree = grid_search.best_estimator_
opt_params = grid_search.best_params_
y_pred = opt_tree.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"best_tree_acc_score = {accuracy_score(y_pred=y_pred, y_true=y_test)}")
print(f"best tree params:", opt_params)

import matplotlib.pyplot as plt
plt.figure()
plot_tree(opt_tree,rounded=True,filled=True, class_names=["0", "1"], fontsize= 10)
plt.tight_layout()
plt.savefig('figs.optimized_tree.png')
plt.show()

base_tree = DecisionTreeClassifier(**opt_tree.get_params())
n_splits = 1000
subset_size = 100

from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone

rs = ShuffleSplit(n_splits=n_splits,train_size=subset_size, random_state= 42)

all_test_preds  = []
per_tree_acc = []
params = opt_tree.get_params()
params.pop("max_features", None)
params["max_features"] = 'sqrt'
base_tree_rf = DecisionTreeClassifier(**params)

for train_idx,_ in rs.split(X_train):

    X_sub, y_sub = X_train[train_idx], y_train[train_idx]
    tree = clone(base_tree_rf)
    tree.fit(X_sub,y_sub)
    pred = tree.predict(X_test)
    all_test_preds.append(pred)
    per_tree_acc.append(accuracy_score(y_pred=pred,y_true=y_test))

import numpy as np
all_test_preds = np.vstack(all_test_preds)
mean_tree_acc = np.mean(per_tree_acc)

print(f"Mean single-tree accuracy over {n_splits} subset-trained trees: {mean_tree_acc:.4f}")

from scipy.stats import mode
y_vote = mode(all_test_preds, axis=0, keepdims=False).mode  # shape: (n_test,)
ensemble_acc_score = accuracy_score(y_pred=y_vote, y_true=y_test)
print(f"rf_acc_score = {ensemble_acc_score:.4f}")

n_trees = 1000
import random
rf_all_pred = []
rf_acc = []
for _ in range(n_trees):
    train_idx = random.choices(range(len(X_train)), k= len(X_train))
    X_boot = X_train[train_idx]
    y_boot = y_train[train_idx]
    tree = clone(base_tree_rf)
    tree.fit(X_boot,y_boot)
    pred = tree.predict(X_test)
    rf_all_pred.append(pred)
    rf_acc.append(accuracy_score(y_pred=pred, y_true=y_test))
    
print(f"Mean single tree accuracy in bootstrap = {np.mean(rf_acc):.4f}")
rf_all_pred = np.vstack(rf_all_pred)
y_majority = mode(rf_all_pred, axis=0, keepdims= False).mode
print(f"random forest accuracy = {accuracy_score(y_pred=y_majority, y_true=y_test)}")
