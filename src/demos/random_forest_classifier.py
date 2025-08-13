from sklearn.ensemble import RandomForestClassifier
from utils import moons_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path

class RandomForestDemo():
    def __init__(self, n_estimators, max_leaf_nodes, n_samples = 1000, noise = 0.2, state= 42) -> None:
        self.n_samples = n_samples
        self.noise = noise
        self.state = state 
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.clf = None
        self.splits_ = None

    def fit(self,):
        X,y = moons_dataset(n_samples=self.n_samples,noise=self.noise,state=self.state)
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=self.state)
        self.splits_ = (X_train, X_test, y_train, y_test)

        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          max_leaf_nodes=self.max_leaf_nodes,
                                          random_state=self.state)
        self.clf.fit(X_train,y_train)
        preds = self.clf.predict(X_test)
        print(f"Random forest accuracy score = {accuracy_score(y_true=y_test, y_pred=self.clf.predict(X_test))}")
        return self
    
    def predict(self,X):
        if self.clf is None:
            raise RuntimeError(f"Classifier not found. Run fit() method before predict()")
        return self.clf.predict(X)
    
    def score(self,):
        if self.clf is None or self.splits_ is None:
            raise RuntimeError(f"Classifier not found. Run fit() method before score()")
        _, X_test, _, y_test = self.splits_
        return accuracy_score(y_true=y_test,y_pred=self.predict(X_test))
    
    def plot_decision_boundary(self, X=None, y=None, padding=0.5, step=0.02):
        if X is None:
            if self.splits_ is None:
                raise RuntimeError("No data to plot. Provide X, y or call fit().")
            _, X, _, y = self.splits_

        # Bounds
        x_min = float(X[:, 0].min()) - padding
        x_max = float(X[:, 0].max()) + padding
        y_min = float(X[:, 1].min()) - padding
        y_max = float(X[:, 1].max()) + padding

        # Mesh grid
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step, dtype=float),
            np.arange(y_min, y_max, step, dtype=float),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # ✅ Always use *your* predict method
        Z = self.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.25)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k', linewidths=0.3)
        plt.title("Random Forest Decision Boundary")
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.tight_layout()

        Path("figs").mkdir(exist_ok=True)
        plt.savefig("figs/decision_boundary.png", dpi=150)
        plt.show()


if __name__ == '__main__':

    rf_clf = RandomForestDemo(n_estimators= 500,max_leaf_nodes=10,)
    rf_clf.fit()
    rf_clf.plot_decision_boundary()

