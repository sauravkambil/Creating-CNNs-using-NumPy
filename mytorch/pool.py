import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.out_channels = self.A.shape[1]
        self.output_size_height = (self.A.shape[2] - self.kernel+ 1)
        self.output_size_width = (self.A.shape[3] - self.kernel+ 1)
        self.Z = np.zeros((len(self.A), self.out_channels,self.output_size_height,self.output_size_width),dtype="float64")
        self.ind_x = np.zeros((self.A.shape[0],self.A.shape[1],self.output_size_height,self.output_size_width))
        self.ind_y = np.zeros((self.A.shape[0],self.A.shape[1],self.output_size_height,self.output_size_width))
        for i in range(self.Z.shape[0]):            #output height
            for j in range(self.Z.shape[1]):        #output width
                for k in range(self.Z.shape[2]):    #channel
                    for l in range(self.Z.shape[3]):
                        self.Z[i,j,k,l] = np.amax(A[i,j, k:k+self.kernel, l:l+self.kernel])
                        ind = np.unravel_index(np.argmax(self.A[i,j,k:k+self.kernel,l:l+self.kernel]), self.A[i,j,k:k+self.kernel,l:l+self.kernel].shape)
                        self.ind_x[i,j,k,l] = ind[0]
                        self.ind_y[i,j,k,l] = ind[1]

        return self.Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape, dtype="float64")
        for i in range(dLdA.shape[0]):
            for j in range(dLdA.shape[1]):
                for k in range(dLdA.shape[2] - self.kernel+1):
                    for l in range(dLdA.shape[3] - self.kernel+1):
                        m = int(self.ind_x[i,j,k,l])
                        n = int(self.ind_y[i,j,k,l])
                        dLdA[i,j,k:k+self.kernel,l:l+self.kernel][m,n] += dLdZ[i,j,k,l]
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.Z = np.zeros((A.shape[0], A.shape[1], A.shape[2] - self.kernel+1, A.shape[3] - self.kernel+1))
        for i in range(self.Z.shape[2]):
            for j in range(self.Z.shape[3]):
                wnd = A[:,:,i:i+self.kernel, j:j+self.kernel]
                self.Z[:,:,i,j] = np.mean(wnd,axis=(2,3)) 
        Z = self.Z
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        self.dLdA = np.zeros(self.A.shape)
        for i in range(dLdZ.shape[2]):
            for j in range(dLdZ.shape[3]):
                self.dLdA[:,:,i:i+self.kernel, j:j+self.kernel] += dLdZ[:,:,i,j].reshape (* self.dLdA.shape[0:2],1,1) * (1/self.kernel**2)
                
        return self.dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        forward = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(forward)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.downsample2d = Downsample2d(self.stride)  # TODO
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)
        return dLdA
