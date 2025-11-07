import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):

    def __init__(self, input_dim: int, output_dim:int, A: torch.Tensor):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.A = A  # Adjacency matrix

        # A_hat = A + I (adding self-loops)
        self.I = torch.eye(self.A.size(0))
        self.A_hat = self.A + self.I

        # Create a degree matrix by summing the values in each row and use the row sum as the diagonal elements to form a diagonal matrix
        self.D_hat = torch.diag(torch.sum(self.A_hat, dim=1))

        # Create a new tensor with diagonal elements and zeros elsewhere
        self.D_hat = torch.diag_embed(torch.diag(self.D_hat))

        # Create D^{-1/2}
        self.D_hat_neg_sqrt = torch.diag_embed(torch.diag(torch.pow(self.D_hat, -0.5)))

        # Initialize the weight matrix as a learnable parameter
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, X: torch.Tensor):
        # Graph convolution operation
        # D^-1/2 * (A_hat * D^-1/2)
        support_1 = torch.matmul(self.D_hat_neg_sqrt, torch.matmul(self.A_hat, self.D_hat_neg_sqrt))

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
    gcn_layer = GCNLayer(input_dim, output_dim, A)

    # Example input feature matrix X (3 nodes with 3 features each)
    X = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]], dtype=torch.float32)

    # Forward pass
    H = gcn_layer(X)

    print("Output feature matrix H:")
    print(H)