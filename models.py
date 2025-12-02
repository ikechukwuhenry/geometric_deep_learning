import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNLayer


class GCN(nn.Module):
    """Graph Convolutional Network Model
    
    Args:
        input_dim (int): Dimension of the input i.e Number of input features per node.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output ie Number of output features per node.
        A (torch.Tensor): Adjacency matrix of the graph (shape: [num_nodes, num_nodes]).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, A: torch.Tensor):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, A)
        self.gcn2 = GCNLayer(hidden_dim, output_dim, A)

    def forward(self, X):
        X = F.relu(self.gcn1(X))
        H = self.gcn2(X)
        # return H
        return F.log_softmax(H, dim=1)
    
# Example usage:
if __name__ == "__main__":
    # Define a simple graph with 3 nodes
    input_dim = 3  # Number of input features per node
    hidden_dim = 4  # Number of hidden features
    output_dim = 2  # Number of output features per node
    A = torch.tensor([[1., 0., 0.],
                      [0., 1., 1.],
                      [0., 1., 1.]], dtype=torch.float32)

    # Create GCN model
    gcn_model = GCN(input_dim, hidden_dim, output_dim, A)


    # alternatively, you can directly test the GCNLayer as follows:
    print("Testing GCN Model Layers:\n")
    for idx, module in enumerate(gcn_model.children()):
        print(f"Layer {idx}: {module}")

    
    for name, module in gcn_model.named_children():
        print(f"Module Name: {name} \nModule Details: {module}")
    
    print("\n\n")

    # Example input feature matrix X (3 nodes with 3 features each)
    X = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]], dtype=torch.float32)

    # Forward pass
    output = gcn_model(X)
    print(output)