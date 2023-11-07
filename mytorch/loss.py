import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        temp = np.ones_like(A[:,0])
        temp2 = np.ones_like(A[0,:])
        self.N = temp.reshape(-1,1)  # TODO
        self.C = temp2.reshape(-1,1)  # TODO
        self.Nlength = A.shape[0]
        self.Clength = A.shape[1]

        se = (self.A-self.Y)*(self.A-self.Y)   # TODO
        sse = np.dot(np.dot((np.transpose(self.N)),se),self.C)  # TODO
        mse = sse/(2*self.Nlength*self.Clength)  # TODO  

        return np.float64(mse)

    def backward(self):

        
        temp = np.ones_like(self.A[:,0])
        temp2 = np.ones_like(self.A[0,:])
        self.N = temp.reshape(-1,1)  # TODO
        self.C = temp2.reshape(-1,1)  # TODO
        self.Nlength = self.A.shape[0]
        self.Clength = self.A.shape[1]

        dLdA =(self.A-self.Y)/(self.Nlength*self.Clength)

        return np.float64(dLdA)


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        #N = None  # TODO
        #C = None  # TODO
        temp = np.ones_like(A[:,0])
        temp2 = np.ones_like(A[0,:])
        self.N = temp.reshape(-1,1)  # TODO
        self.C = temp2.reshape(-1,1)  # TODO
        self.Nlength = A.shape[0]
        self.Clength = A.shape[1]
        #Ones_C = None  # TODO
        #Ones_N = None  # TODO

        self.softmax = np.exp(self.A)/np.sum(np.exp(self.A), axis=1, keepdims=True)  # TODO
        crossentropy = np.dot(-self.Y * np.log(self.softmax),self.C)  # TODO
        sum_crossentropy = np.dot(np.transpose(self.N),crossentropy)  # TODO
        L = sum_crossentropy / self.Nlength

        return np.float64(L)

    def backward(self):

        dLdA = np.exp(self.A)/np.sum(np.exp(self.A), axis=1, keepdims=True) - self.Y  # TODO

        return np.float64(dLdA)
