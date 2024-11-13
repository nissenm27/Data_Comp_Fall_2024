import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Device configuration
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Step 1: Determine Feature Importances
data = pd.read_csv("ecological_health_dataset.csv")
data = data.drop(columns=['Timestamp'])
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())
valid_labels = [1, 2, 3, 4]
data = data[data['Ecological_Health_Label'].isin(valid_labels)]
data = data.reset_index(drop=True)
X = data.drop(columns=['Ecological_Health_Label']).values
y = data['Ecological_Health_Label'].astype(int).values - 1

# Split data for Random Forest
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train_rf)

# Get feature importances
importances = rf.feature_importances_
feature_names = data.columns[:-1]

# Create DataFrame of feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

# Select top N features
N = 8
top_features = feature_importance_df['Feature'].head(N).tolist()

# Step 2: Update Dataset Class
class EnvironmentalDataset(Dataset):
    def __init__(self, csv_file, selected_features=None):
        # Load the data
        self.data = pd.read_csv(csv_file)

        # Drop non-numeric columns such as 'Timestamp'
        self.data = self.data.drop(columns=['Timestamp'])

        # Convert all columns to numeric
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        self.data = self.data.fillna(0)

        # Filter out invalid labels
        valid_labels = [1, 2, 3, 4]
        self.data = self.data[self.data['Ecological_Health_Label'].isin(valid_labels)]
        self.data = self.data.reset_index(drop=True)

        # If selected_features is provided, select those features
        if selected_features is not None:
            self.data = self.data[selected_features + ['Ecological_Health_Label']]

        # Normalize the features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.data.iloc[:, :-1].values)

        # Adjust labels
        self.labels = self.data.iloc[:, -1].astype(int).values - 1

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

# Instantiate the dataset with selected features
dataset = EnvironmentalDataset("ecological_health_dataset.csv", selected_features=top_features)

# Update input_size
input_size = len(top_features)

# Split the dataset indices
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=dataset.labels,
    random_state=42
)

# Create training and validation subsets
train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=100, shuffle=False)

# Recompute class weights using training labels
train_labels = [dataset.labels[i] for i in train_indices]
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define the EcosystemHealthPredictor model
class EcosystemHealthPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(EcosystemHealthPredictor, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.5))
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Model parameters
hidden_sizes = [128, 64, 32, 16]
output_size = 4

# Instantiate the model and move it to the device
model = EcosystemHealthPredictor(input_size, hidden_sizes, output_size).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=50):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)

        # Validation Phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Step the scheduler
        scheduler.step(avg_val_loss)

# Train the model
train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=100)
