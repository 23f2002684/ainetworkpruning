import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns to prune its own weights dynamically during training.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # The learnable gate parameters
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the weights, biases, and gate scores."""
        # Initialize standard weights and bias similarly to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate scores to positive values so initial gates are active (around 0.73)
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, input):
        """
        Calculates pruned weights using a sigmoid transformation on gate scores,
        then applies the standard linear transformation.
        """
        # Apply sigmoid to squash gate scores between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication to prune weights
        pruned_weights = self.weight * gates
        
        # Standard linear projection
        return F.linear(input, pruned_weights, self.bias)


class PrunableNet(nn.Module):
    """
    A Feed-Forward Neural Network using the custom PrunableLinear layers for CIFAR-10.
    """
    def __init__(self):
        super(PrunableNet, self).__init__()
        # CIFAR-10 images are 3 channels of 32x32 pixels = 3072 input features
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer (CrossEntropyLoss handles it)
        return x


def calculate_sparsity_loss(model):
    """
    Calculates the L1 norm of all gates in the network.
    This encourages the model to turn gates off (towards 0).
    """
    sparsity_loss = 0.0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            # The actual gate values are sigmoid(gate_scores)
            gates = torch.sigmoid(m.gate_scores)
            # Add L1 norm (sum of absolute values, which are just sum since gates > 0)
            sparsity_loss += torch.sum(gates)
    return sparsity_loss


def calculate_sparsity_level(model, threshold=1e-2):
    """
    Calculates the percentage of weights that are effectively pruned (gate < threshold).
    """
    total_weights = 0
    pruned_weights = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                total_weights += gates.numel()
                pruned_weights += torch.sum(gates < threshold).item()
    
    return (pruned_weights / total_weights) * 100.0 if total_weights > 0 else 0.0


def train(model, device, train_loader, optimizer, epoch, lambda_val):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        
        # Standard classification loss
        classification_loss = criterion(output, target)
        
        # Sparsity regularization loss
        sparsity_loss = calculate_sparsity_loss(model)
        
        # Total combined loss
        loss = classification_loss + lambda_val * sparsity_loss
        loss.backward()
        
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Class Loss: {classification_loss.item():.4f}\t'
                  f'Sparsity Loss: {sparsity_loss.item():.4f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


def plot_gate_distribution(model, lambda_val):
    """
    Plots the histogram of the gate values for the best model 
    to visualize the distribution of pruning.
    """
    all_gates = []
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                all_gates.extend(gates.cpu().numpy().flatten())
                
    plt.figure(figsize=(8, 5))
    plt.hist(all_gates, bins=50, alpha=0.75, color='teal')
    plt.title(f'Gate Value Distribution (λ = {lambda_val})')
    plt.xlabel('Gate Value (Sigmoid Output)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'gate_distribution_lambda_{lambda_val}.png')
    plt.close()
    print(f"Saved distribution plot to gate_distribution_lambda_{lambda_val}.png")


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple normalizations for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # We will test a small, medium, and large lambda mapping to sparsity constraints
    lambdas = [1e-6, 1e-5, 1e-4]
    epochs = 5 # Reduced for quicker demonstration, adjust upwards for better results
    
    results = []
    best_model = None
    best_lambda = None
    
    for l in lambdas:
        print(f"\n{'='*40}\nTraining with λ = {l}\n{'='*40}")
        model = PrunableNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, l)
            
        test_acc = test(model, device, test_loader)
        sparsity = calculate_sparsity_level(model)
        
        print(f"Final Sparsity for λ {l}: {sparsity:.2f}%")
        results.append((l, test_acc, sparsity))
        
        plot_gate_distribution(model, l)

    print("\n" + "="*52)
    print("--- Summary of Results ---")
    print(f"{'Lambda':<10} | {'Test Accuracy (%)':<18} | {'Sparsity Level (%)':<18}")
    print("-" * 52)
    for l, acc, sp in results:
        print(f"{l:<10} | {acc:<18.2f} | {sp:<18.2f}")
    print("="*52 + "\n")


if __name__ == '__main__':
    main()
