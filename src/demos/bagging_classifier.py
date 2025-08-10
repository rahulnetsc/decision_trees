from scipy.stats import mode
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from utils import moons_dataset

class Bagging_Classifier():
    def __init__(self,n_samples, noise, state = 42):
        self.n_samples = n_samples
        self.noise = noise
        X,y = moons_dataset(n_samples=self.n_samples,noise=self.noise, state=state)

    def fit(self,):
        pass




