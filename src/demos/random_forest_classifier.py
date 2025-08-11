from sklearn.ensemble import RandomForestClassifier
from utils import moons_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class RandomForest():
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
        return accuracy_score(y_true=y_test,y_pred=self.clf.predict(X_test))
    
    def plot_decision_boundary(self,clf=None,X=None,y=None, padding =0.5, step= 0.02):

        if clf is None:
            if self.clf is None:
                raise RuntimeError(f"Classifier not found. Run fit() or pass clf")
            clf = self.clf
        if X is None or y is None:
            if self.splits_ is None:
                raise RuntimeError(f"no data to plot")
            _,X,_,y = self.splits_
        
        x_min, x_max = X[:,0].min()-padding, X[:,0].max()+padding
        y_min, y_max = X[:,1].min()-padding, X[:,1].max()+padding

        xx,yy = np.meshgrid(
            np.arange(x_min,x_max,step),
            np.arange(y_min,y_max,step),
        )

        grid = np.c_[xx.ravel(),yy.ravel()]
        Z = clf.predict(grid).reshape(xx.shape)
        plt.contourf(xx,yy,Z, alpha = 0.25)
        plt.scatter(X[:,0],X[:,1], c=y, s=20, edgecolors= 'k')
        plt.title(f"Random forest classifier decision boundary")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()
        plt.savefig(f"figs/random_forest_classifier.png")
        plt.show(block= False)

if __name__ == '__main__':

    rf_clf = RandomForest(n_estimators= 500,max_leaf_nodes=10,)
    rf_clf.fit()
    rf_clf.plot_decision_boundary()





