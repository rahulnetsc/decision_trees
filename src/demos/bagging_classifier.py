from scipy.stats import mode
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import moons_dataset
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class Bagging_Classifier():
    def __init__(self,n_samples, noise, max_samples = 0.8, state = 42, base_estimator = None):
        self.n_samples = n_samples
        self.noise = noise
        self.state = state 
        self.max_samples = max_samples
        self.base_estimator = base_estimator

        self.clf = None
        self.splits_ = None

    def fit(self,):
        
        X,y = moons_dataset(n_samples=self.n_samples,noise=self.noise, state=self.state)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state= self.state)
        self.splits_ = (X_train, X_test, y_train, y_test)

        base = self.base_estimator or DecisionTreeClassifier(random_state=self.state)
        self.clf = BaggingClassifier(estimator=base, max_samples= self.max_samples, 
                                n_estimators= 500, random_state= self.state,n_jobs=-1)
        self.clf.fit(X=X_train,y=y_train)
        preds = self.clf.predict(X_test)
        print(f"accuracy_score = {accuracy_score(y_pred=preds,y_true=y_test):.3f}")
        return self
    
    def predict(self,X):
        if self.clf is None:
            raise RuntimeError(f"Run fit method before predict")
        return self.clf.predict(X)
    
    def score(self,):
        if self.clf is None or self.splits_ is None:
            raise RuntimeError(f"Run fit method before predict")
        X_train, X_test, y_train, y_test = self.splits_
        return accuracy_score(y_true=y_test, y_pred=self.clf.predict(X_test))
    
    def plot_decision_boundary(self, X=None, y=None, clf=None, padding = 0.5, step=0.02):
        ''' Plot decision boundary of classifier for X,y'''
        if clf is None:
            if self.clf is None:
                raise RuntimeError(f"No classifier to plot. Pass clf or call fit method first.")
            clf = self.clf
        if X is None or y is None:
            if self.splits_ is None:
                raise RuntimeError(f"No data to plot")
            _,X,_,y = self.splits_

        x_min, x_max = X[:,0].min()-padding, X[:,0].max()+padding
        y_min, y_max = X[:,1].min()-padding, X[:,1].max()+padding
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max,step),
            np.arange(y_min,y_max,step)
        )
        grid = np.c_[xx.ravel(),yy.ravel()]
        Z = clf.predict(grid).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.25)
        plt.scatter(X[:,0], X[:,1], c=y, s=20, edgecolor='k')
        plt.title("Bagging decision boundary")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()
        plt.savefig('figs/bagging_classifier.png')
        plt.show()
        
if __name__ == '__main__':
    bag_clf = Bagging_Classifier(n_samples= 1000, noise= 0.2)
    bag_clf.fit()
    bag_clf.plot_decision_boundary()




