import torch
import torch.nn as nn
import torch.optim as optim


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
output_size = 15  # The number of initial features for now...

model = EcosystemHealthPredictor(input_size, hidden_sizes, output_size)

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

# Assuming data_loader is your DataLoader for EnvironmentalData batches
