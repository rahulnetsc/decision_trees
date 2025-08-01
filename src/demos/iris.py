from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time 
import numpy as np
from graphviz import Source

X,y = load_iris(return_X_y=True, as_frame=False)
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
tree_clf =  DecisionTreeClassifier()
start_time = time.time()
tree_clf.fit(X=X_train,y=y_train)
y_pred = tree_clf.predict(X=X_test)
score = accuracy_score(y_pred=y_pred, y_true=y_test)

print(f"Time: {(time.time()-start_time)/60:.2f} mins, acc_score: {score:.2f}")
print(f"class_labels: {np.unique(y)}, feature_names = {load_iris().feature_names}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plot_tree(tree_clf,
        feature_names=load_iris().feature_names,
        class_names=[str(i) for i in np.unique(y)],
        rounded = True, filled=True,
        )
plt.title("Decision Tree (Depth=2)")
plt.savefig("src/demos/iris_tree.png")  # Optional: Save the plot
plt.show()

