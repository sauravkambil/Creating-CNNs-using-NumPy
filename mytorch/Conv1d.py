# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels, dtype='float64')
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape, dtype='float64')
        self.dLdb = np.zeros(self.b.shape, dtype='float64')

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        self.A = A
        
        self.output_size = ((len(self.A[0][1]) - len(self.W[0][1]))+ 1)
        self.Z = np.zeros((len(self.A), self.out_channels,self.output_size),dtype="float64")
       
        for i in range(self.Z.shape[2]):
            wnd = self.A[:,:,i:i+self.kernel_size]
            self.Z[:,:,i] = np.tensordot(wnd,self.W,axes=((1,2),(1,2))) + self.b
        
        return np.float64(self.Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        batch_size, _, output_size = dLdZ.shape
        input_size = self.A.shape[2]

        self.dLdW = np.zeros_like(self.W, dtype='float64')
        self.dLdb = np.sum(dLdZ, axis=(0, 2))
        dLdA = np.zeros_like(self.A, dtype='float64')

    # Compute dLdW
        for i in range(self.kernel_size):
            dLdW_slice = self.A[:, :, i:i+output_size]
            self.dLdW[:,:,i]= np.tensordot(dLdZ, dLdW_slice, axes=((2,0),(2,0)))

    # Compute dLdA
        W_flipped = np.flip(self.W, axis=2)
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), mode='constant')
        for i in range(input_size):
            dLdA[:, :, i] = np.tensordot(dLdZ_padded[:,:,i:i+W_flipped.shape[2]], W_flipped, axes = ((1,2),(0,2)))

        # self.dLdW = None  # TODO
        # self.dLdb = None  # TODO
        # dLdA = None  # TODO

        return np.float64(dLdA)


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W = weight_init_fn
        self.b = bias_init_fn
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size) # TODO
        self.downsample1d = Downsample1d(stride)  # TODO
        print('1')

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        # Call Conv1d_stride1
        conn = self.conv1d_stride1.forward(A)
        # downsample
        Z = self.downsample1d.forward(conn) # TODO
        return np.float64(Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        back = self.downsample1d.backward(dLdZ)
        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(back)  # TODO

        return np.float64(dLdA)
