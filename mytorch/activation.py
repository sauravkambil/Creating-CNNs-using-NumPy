import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="float64")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1/(1+np.exp(-Z))

        return np.float64(self.A)

    def backward(self):

        dAdZ = self.A - self.A*self.A

        return np.float64(dAdZ)


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        self.A = np.tanh(Z)

        return np.float64(self.A)

    def backward(self):
        
        dAdZ= np.ones(self.A.shape, dtype="float64")-np.square(self.A)

        return np.float64(dAdZ)

class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    

    def forward(self, Z):

        self.A = np.array(np.maximum(0.0,Z), dtype='float64')

        return np.float64(self.A)

    def backward(self):
        dAdZ=self.A
        dAdZ[dAdZ>0]=1
        dAdZ[dAdZ<=0]=0

        return np.float64(dAdZ)