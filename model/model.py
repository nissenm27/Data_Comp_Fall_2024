import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# This is for my mac so I can run it on my gpu
# Check if the Mac GPU (MPS) is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Custom Dataset class
class EnvironmentalDataset(Dataset):
    def __init__(self, csv_file):
        # Load the data
        self.data = pd.read_csv(csv_file)
        # Separate features and labels
        self.features = self.data.iloc[:, 1:-1].values  # Assume all columns except last are features
        self.labels = self.data.iloc[:, -1].values      # Assume last column is the label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert features and labels to PyTorch tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

# DataLoader function
def create_data_loader(csv_file, batch_size):
    dataset = EnvironmentalDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Defining a class EcosystemHealthPredictor
class EcosystemHealthPredictor(nn.Module):
    # init is what is passed into a neuron
    def __init__(self, input_size, hidden_sizes, output_size):
        super(EcosystemHealthPredictor, self).__init__()
        layers = []
        last_size = input_size
        # Iterating through size of the hidden layers
        for hidden_size in hidden_sizes: 
            layers.append(nn.Linear(last_size, hidden_size)) 
            layers.append(nn.ReLU()) # Activation Function
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)
    # forward is what the neuron is sending the next layer
    def forward(self, x):
        return self.network(x)

# Assuming 15 features and 1 output for ecosystem health score or label
input_size = 15 # Features
hidden_sizes = [64, 32, 16] # First hidden layer is 64 neurons, second is 32 neurons, third is 16 neurons
output_size = 1  # Just predict the environment's overall health for now

# Instantiate the model and move it to the appropriate device
model = EcosystemHealthPredictor(input_size, hidden_sizes, output_size).to(device)

# Example training loop
criterion = nn.MSELoss()  # Use nn.CrossEntropyLoss() for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data_loader, epochs=50):
    model.train()
    for epoch in range(epochs):
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Usage
data_loader = create_data_loader("ecological_health_dataset.csv", batch_size=100)
train_model(model, data_loader, epochs=50)