from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from utils import moons_dataset
import matplotlib.pyplot as plt
from pathlib import Path

class AdaBoost():  

    def __init__(self,n_estimators: int, lr = 0.5) -> None:
        self.n_estimators = n_estimators
        self.lr = lr
        self.splits_ = None
        self.classes_ = None
        self.clf = None
        self.confidence = None

    
    def fit(self):
        
        X,y = moons_dataset()   
        X_train,X_test,y_train,y_test = train_test_split(X,y)
        self.splits_ = (X_train,X_test,y_train,y_test)
        self.classes_ = np.unique(y_train)
        eps = 0.01

        m = len(X_train)
        tree: DecisionTreeClassifier = DecisionTreeClassifier()
        predictors = []
        predictor_weights = []
        weight = 1/m* np.ones(m)
        weight_error_rate = []
        for j in range(self.n_estimators):
            predictor = clone(tree)
            predictor.fit(X_train,y_train, sample_weight=weight)
            predictors.append(predictor)

            preds = predictor.predict(X_train)
            index = (preds!= y_train)
            error_rate = sum(weight[index])
            weight_error_rate.append(error_rate)
            alpha_pred = self.lr * np.log((1-error_rate)/(error_rate+eps))
            predictor_weights.append(alpha_pred)
            weight[index] = weight[index]*np.exp(alpha_pred)
            weight /= sum(weight)

        self.clf = predictors
        self.confidence = predictor_weights
        return self
    
   

    def weighted_votes(self,preds_mat: np.ndarray, alphas: np.ndarray, classes: np.ndarray):
        """
        preds_mat: (T, n) integer labels predicted by each tree
        alphas:    (T,)    weight per tree
        classes:   (K,)    unique class labels (sorted)
        returns:
        scores:  (K, n)  weighted votes per class for each sample
        """
        # mask[k, t, i] = 1 iff tree t predicted class classes[k] for sample i
        mask = (preds_mat[None, :, :] == classes[:, None, None])  # (K, T, n) boolean
        # weight each tree’s votes by alpha and sum over trees
        scores = (mask * alphas[None, :, None]).sum(axis=1)       # (K, n)
        return scores

    def predict_from_votes(self,preds_mat, alphas, classes):
        scores = self.weighted_votes(preds_mat, alphas, classes)       # (K, n)
        y_pred = classes[np.argmax(scores, axis=0)]               # (n,)
        return y_pred

    def predict(self,X=None, y= None, clf = None):
        if clf is None:
            if self.clf is None:
                raise RuntimeError(f"No classifier found. Run fit() or pass clf")
            clf = self.clf
        
        if X is None:
            if self.splits_ is None:
                raise RuntimeError(f"No data found")
            _,X,_,y = self.splits_
        if self.classes_ is None:
            raise RuntimeError(f"Class labels not found. run fit() first")
        classes = self.classes_
        all_preds = []
        for tree in clf:
            preds = tree.predict(X)
            all_preds.append(preds)
        all_preds = np.vstack(all_preds) #(num_trees, num_instances)
        if self.confidence is None:
            raise RuntimeError(f"Confidence scores not found. Run fit()")
        alphas = np.asarray(self.confidence)                                     # (T,)
        
        # Multiclass (SAMME) weighted voting
        mask = (all_preds[None, :, :] == classes[:, None, None])              # (K, T, n)
        scores = (mask * alphas[None, :, None]).sum(axis=1)                   # (K, n)
        return classes[np.argmax(scores, axis=0)]

    def score(self,):
        if self.clf is None or self.splits_ is None:
            raise RuntimeError(f"Classifier not found. Run fit() method before score()")
        _, X_test, _, y_test = self.splits_
        return accuracy_score(y_true=y_test,y_pred=self.predict(X_test))
    
    def plot_decision_boundary(self, X=None, y=None, padding=0.5, step=0.02):
        if X is None or y is None:
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
        plt.title("Custom AdaBoost Decision Boundary")
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.tight_layout()

        Path("figs").mkdir(exist_ok=True)
        plt.savefig("figs/decision_boundary.png", dpi=150)
        plt.show()

if __name__ == '__main__':

    ada_clf = AdaBoost(n_estimators= 100)
    ada_clf.fit()
    ada_clf.predict()
    print(f"Ada Boost accuracy score = {ada_clf.score()}")
    ada_clf.plot_decision_boundary()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       