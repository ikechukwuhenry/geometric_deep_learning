# this was from the tutqorials and codes 
# by J Rosser: https://medium.com/@jrosseruk/demystifying-gcns-a-step-by-step-guide-to-building-a-graph-convolutional-network-layer-in-pytorch-09bf2e788a51
# is an implementation of a graph convolution from this paper: https://arxiv.org/pdf/1609.02907

import torch
import torch.nn as nn
import torch.nn.functional as F

class GConvLayer(nn.Module):
    """Graph Convolutional Layer as described in Kipf & Welling (2016)
    
    Args:
        input_dim (int): Dimension of the input i.e  Number of input features per node.
        output_dim (int): Dimension of the output ie Number of output features per node.
        A (torch.Tensor): Adjacency matrix of the graph (shape: [num_nodes, num_nodes]).
    """

    def __init__(self, input_dim: int, output_dim:int, A: torch.Tensor):
        super(GConvLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A  # Adjacency matrix

        # A_hat = A + I (adding self-loops)
        self.I = torch.eye(A.size(0))
        self.A_hat = self.A + self.I

        # Create diagonal degree matrix
        self.ones = torch.ones(input_dim, input_dim)
        self.D = torch.matmul(self.A.float(), self.ones.float())

        # Extract the diagonal elements to form a diagonal matrix
        self.D = torch.diag(self.D)


        # line 37 is equivalent to: line 29 - 33
        # self.D = torch.diag(torch.sum(self.A_hat, dim=1))

        # Create a new tensor with diagonal elements and zeros elsewhere
        self.D = torch.diag_embed(self.D)

        # Create D^{-1/2}
        self.D_neg_sqrt = torch.diag_embed(torch.diag(torch.pow(self.D, -0.5)))


        # Initialize the weight matrix as a learnable parameter
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, X: torch.Tensor):

        # Graph convolution operation
        # D^-1/2 * (A_hat * D^-1/2)
        support_1 = torch.matmul(self.D_neg_sqrt, torch.matmul(self.A_hat, self.D_neg_sqrt))

        # (D^-1/2 * A_hat * D^-1/2) * (X * W)
        support_2 = torch.matmul(support_1, torch.matmul(X, self.W))

        # Relu(D^-1/2 * A_hat * D^-1/2 * X * W)
        H = F.relu(support_2)

        return H
    

# Example usage:
if __name__ == "__main__":
    # Define a simple graph with 3 nodes
    input_dim = 3 # Number of input features per node ie assuming the input dimension is 3
    output_dim = 2 # Number of output features per node ie assuming the output dimension is 2
    A = torch.tensor([[1., 0., 0.],
                      [0., 1., 1.],
                      [0., 1., 1.]], dtype=torch.float32)

    # Create GCN layer
    gcn_layer = GConvLayer(input_dim, output_dim, A)

    # Example input feature matrix X (3 nodes with 3 features each)
    X = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]], dtype=torch.float32)

    # Forward pass
    H = gcn_layer(X)

    print("Output feature matrix H:")
    print(H)