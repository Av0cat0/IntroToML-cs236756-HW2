from sklearn.base import BaseEstimator, ClassifierMixin
import scipy
import numpy as np
class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 5):
        self.X_y = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_y=(X,y)
        return self

    def predict(self, X):
        # Note: You can use self.n_neighbors here
        predictions = None
        # TODO: compute the predicted labels (+1 or -1)
        dist = scipy.spatial.distance.cdist(X, self.X_y[0], 'euclidean')
        x_neighbors = np.argpartition(dist, self.n_neighbors)[:,:self.n_neighbors]
        y_pred=np.take(self.X_y[1],x_neighbors)
        predictions=scipy.stats.mode(y_pred, axis=1)[0]
        return predictions