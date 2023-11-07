import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        output_width = self.upsampling_factor*(input_width-1) +1
        #A_upsampled = np.repeat(A, self.upsampling_factor, axis=-1)
        A_upsampled = np.zeros((batch_size, in_channels, int(output_width)), dtype='float64')
        #A_upsampled = np.reshape(A_upsampled, (batch_size, in_channels, output_width))
        A_upsampled[:, :, ::self.upsampling_factor] = A

        Z = A_upsampled  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        k = self.upsampling_factor
        batch_size, in_channels, output_width = dLdZ.shape
        input_width = (output_width - 1) // k + 1

        # Downsample dLdZ
        dLdZ_downsampled = dLdZ[:, :, ::k]

        dLdA = dLdZ_downsampled  # TODO

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        k = self.downsampling_factor
        output_width = (input_width) // k + 1
        Z = A[:, :, ::k][:, :, :output_width] 
        #Z = A[:, :, ::k] # TODO
        self.input_width_back = input_width
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        k = self.downsampling_factor
        batch_size, in_channels, output_width = dLdZ.shape
        input_width = k * (output_width-1) +1

        dLdA = np.zeros((batch_size, in_channels, self.input_width_back), dtype='float64')
        dLdA[:, :, ::k] = dLdZ  # TODO

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        batch_size, in_channels, input_width, input_height = A.shape

        # Calculate output dimensions
        output_width = (input_width-1) * self.upsampling_factor +1
        output_height = (input_height-1) * self.upsampling_factor +1

        # Initialize output tensor
        Z = np.zeros((batch_size, in_channels, output_width, output_height), dtype='float64')

        # Copy input tensor to output tensor, upsampling the dimensions
        for i in range(input_width):
            for j in range(input_height):
                Z[:, :, i * self.upsampling_factor, j * self.upsampling_factor] = A[:, :, i, j]  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        batch_size, in_channels, output_width, output_height = dLdZ.shape

        # Calculate input dimensions
        input_width = int(np.ceil(output_width /self.upsampling_factor))
        input_height = int(np.ceil(output_height /self.upsampling_factor))

        # Initialize output tensor
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height), dtype='float64')

        # Copy relevant elements of output tensor to input tensor, downsampling the dimensions
        for i in range(input_width):
            for j in range(input_height):
                row_idx = min(i * self.upsampling_factor, output_width - 1)
                col_idx = min(j * self.upsampling_factor, output_height - 1)
                dLdA[:, :, i, j] = dLdZ[:, :, row_idx, col_idx]  # TODO

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        batch_size, in_channels, input_width, input_height = A.shape
        #output_width = input_width // self.downsampling_factor +1
        #output_height = input_height // self.downsampling_factor +1
        output_width = int(np.ceil((input_width) / self.downsampling_factor))
        output_height = int(np.ceil((input_height) / self.downsampling_factor))
        self.input_width_back = input_width
        self.input_height_back = input_height
        # Initialize the output tensor
        Z = np.zeros((batch_size, in_channels, output_width, output_height), dtype='float64')

        # Downsample the input tensor by taking every downsampling_factor-th element
        for i in range(output_width):
            for j in range(output_height):
                row_idx = min(i * self.downsampling_factor, input_width - 1)
                col_idx = min(j * self.downsampling_factor, input_height - 1)
                Z[:, :, i, j] = A[:, :, row_idx, col_idx]# TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = output_width * self.downsampling_factor
        input_height = output_height * self.downsampling_factor
    
        # Initialize the output tensor
        dLdA = np.zeros((batch_size, in_channels, self.input_width_back, self.input_height_back), dtype='float64')

        # Upsample the gradient tensor by filling in the extra elements with zeros
        for i in range(output_width):
            for j in range(output_height):
                row_idx = min(i * self.downsampling_factor, self.input_width_back - 1)
                col_idx = min(j * self.downsampling_factor, self.input_height_back - 1)
                dLdA[:, :, row_idx, col_idx] = dLdZ[:, :, i, j]  # TODO

        return dLdA
