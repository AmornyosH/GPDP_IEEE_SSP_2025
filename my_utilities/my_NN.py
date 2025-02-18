import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        num_nodes = 256
        self.fc1 = torch.nn.Linear(input_dim, num_nodes)
        self.fc2 = torch.nn.Linear(num_nodes, num_nodes)
        self.fc3 = torch.nn.Linear(num_nodes, num_nodes)
        self.output = torch.nn.Linear(num_nodes, output_dim)

    def forward(self, input):
        act1 = torch.nn.functional.mish(self.fc1(input))
        act2 = torch.nn.functional.mish(self.fc2(act1))
        act3 = torch.nn.functional.mish(self.fc3(act2))
        # act1 = torch.nn.functional.relu(self.fc1(input))
        # act2 = torch.nn.functional.relu(self.fc2(act1))
        # act3 = torch.nn.functional.relu(self.fc3(act2))
        output = self.output(act3)
        return output
