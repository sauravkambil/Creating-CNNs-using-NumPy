import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels, dtype='float64')
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape, dtype='float64')
        self.dLdb = np.zeros(self.b.shape, dtype='float64')

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel_size + 1
        output_height = input_height - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_width, output_height), dtype='float64')
        for i in range(output_width):
            for j in range(output_height):
                w = self.A[:,:,i:i+self.kernel_size,j:j+self.kernel_size]
                Z[:,:,i,j] = np.tensordot(w, self.W, axes=((1,2,3),(1,2,3))) + self.b
        

        #Z = None  # TODO

        return np.float64(Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, _, output_width, output_height = dLdZ.shape
        input_width, input_height = self.A.shape[2], self.A.shape[3]

        self.dLdW = np.zeros_like(self.W, dtype='float64')
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        dLdA = np.zeros_like(self.A, dtype='float64')

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dLdW_slice = self.A[:, :, i:i+output_width, j:j+output_height]
                self.dLdW[:, :, i, j] = np.tensordot(dLdZ, dLdW_slice, axes=((0, 2, 3), (0, 2, 3)))

        # Compute dLdA
        W_flipped = np.flip(self.W, axis=(2, 3))
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)), mode='constant')
        for i in range(input_width):
            for j in range(input_height):
                dLdA[:, :, i, j] = np.tensordot(dLdZ_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size], W_flipped, axes=((1,2,3),(0,2,3)))

        # self.dLdW = None  # TODO
        # self.dLdb = None  # TODO
        # dLdA = None  # TODO

        return np.float64(dLdA)


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W = weight_init_fn
        self.b = bias_init_fn
        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        # TODO
        self.A = A
        # Call Conv2d_stride1
        conn = self.conv2d_stride1.forward(A)
        # downsample
        Z = self.downsample2d.forward(conn)

        # downsample
        #Z = None  # TODO

        return np.float64(Z)

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        back = self.downsample2d.backward(dLdZ)
        # Call Conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(back)  # TODO

        return np.float64(dLdA)
