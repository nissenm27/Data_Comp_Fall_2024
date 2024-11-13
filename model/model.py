import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Device configuration (use GPU if available)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Custom Dataset class for environmental data
class EnvironmentalDataset(Dataset):
    def __init__(self, csv_file):
        # Load the data
        self.data = pd.read_csv(csv_file)

        # Drop non-numeric columns such as 'Timestamp'
        self.data = self.data.drop(columns=['Timestamp'])

        # Convert all columns to numeric (coerce errors in case of stray non-numeric values)
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        
        # Fill any NaN values that may have been created
        self.data = self.data.fillna(0)

        # Separate features and labels
        self.features = self.data.iloc[:, :-1].values  # All columns except last are features
        self.labels = self.data.iloc[:, -1].astype(int).values  # Last column is the label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert features and labels to PyTorch tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # For CrossEntropyLoss, labels must be long (integer) type
        return features, label

# DataLoader function
def create_data_loader(csv_file, batch_size):
    dataset = EnvironmentalDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the EcosystemHealthPredictor model
class EcosystemHealthPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(EcosystemHealthPredictor, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes: 
            layers.append(nn.Linear(last_size, hidden_size)) 
            layers.append(nn.ReLU())  # Activation function
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Model parameters
input_size = 14  # Number of features after dropping 'Timestamp'
hidden_sizes = [64, 32, 16]  # Layer sizes
output_size = 4  # Number of unique classes in 'Ecological_Health_Label'

# Instantiate the model and move it to the device
model = EcosystemHealthPredictor(input_size, hidden_sizes, output_size).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data_loader, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Usage
data_loader = create_data_loader("ecological_health_dataset.csv", batch_size=100)
train_model(model, data_loader, epochs=50)
