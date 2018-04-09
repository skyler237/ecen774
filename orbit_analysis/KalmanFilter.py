from IPython.core.debugger import set_trace
import math
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.F = None # State transition
        self.G = None # Input transition
        self.H = None # Measurement model
        self.J = None # Input-measurement model

        self.P = None # State covariance
        self.Q = None # Model noise covariance
        self.R = None # Measurement noise covariance

        self.xhat = None

    def predict(self, u=None):
        if self.F is None or self.P is None or self.Q is None:
            e = "Error: Must initialize matrices (F,P,Q,etc) before running the filter"
            raise Exception(e)
        if u is not None and self.G is None:
            e = "Error: Must initialize G if using an input to the model"
            raise Exception(e)

        # Update state
        if u is None:
            self.xhat = self.F.dot(self.xhat.reshape(-1,1))
        else:
            u = np.reshape(-1,1) # Make sure it is a column vector
            self.xhat = self.F.dot(self.xhat.reshape(-1,1)) + self.G.dot(u)

        # Update covariance
        self.P = self.F.dot(self.P.dot(self.F.T)) + self.Q

        return np.copy(self.xhat)

    def update(self, y):
        if self.H is None or self.P is None or self.R is None:
            e = "Error: Must initialize matrices (H,P,R,etc) before running the filter"
            raise Exception(e)

        # Make sure vectors are columns
        y = np.reshape(y, (-1,1))
        self.xhat = np.reshape(self.xhat, (-1,1))

        # Get residual
        set_trace()
        r = y - self.H.dot(self.xhat)
        # Get innovation covariance
        S = self.R + self.H.dot(self.P.dot(self.H.T))
        # Get kalman gain
        K = self.P.dot(self.H.T.dot(np.linalg.inv(S)))
        # Update state
        self.xhat = self.xhat + K.dot(y)
        # Update covariance
        I = np.eye(np.shape(self.P)[0])
        self.P = (I - K.dot(self.H)).dot(self.P).dot((I - K.dot(self.H)).T) + K.dot(self.R).dot(K.T)

        return np.copy(self.xhat)
