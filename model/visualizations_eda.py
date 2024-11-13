import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("ecological_health_dataset_original.csv")

# Drop non-numeric columns such as 'Timestamp' if present
if 'Timestamp' in data.columns:
    data = data.drop(columns=['Timestamp'])

# Map Pollution_Level to numeric values
pollution_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
if 'Pollution_Level' in data.columns:
    data['Pollution_Level'] = data['Pollution_Level'].map(pollution_mapping)

# Filter out invalid labels for Ecological_Health_Label
valid_labels = ['Ecologically Degraded', 'Ecologically Critical', 'Ecologically Stable', 'Ecologically Healthy']
data = data[data['Ecological_Health_Label'].isin(valid_labels)]
data = data.reset_index(drop=True)

# Convert Ecological_Health_Label to numeric labels for analysis purposes
health_label_mapping = {
    'Ecologically Degraded': 1, 
    'Ecologically Critical': 2, 
    'Ecologically Stable': 3, 
    'Ecologically Healthy': 4
}
data['Ecological_Health_Label'] = data['Ecological_Health_Label'].map(health_label_mapping)

# Get a list of features (exclude the target variable)
features = [col for col in data.columns if col != 'Ecological_Health_Label']

# Loop through each feature and create a boxplot
for feature in features:
    plt.figure(figsize=(10, 6))
    
    # Create the boxplot with customized median line color
    sns.boxplot(
        x='Ecological_Health_Label', 
        y=feature, 
        data=data,
        medianprops={'color': 'red', 'linewidth': 2}  # Customize median line color and thickness
    )
    
    plt.title(f'Boxplot of {feature} by Ecological Health Label')
    plt.xlabel('Ecological Health Label')
    plt.ylabel(feature)
    plt.show()

# Plot histograms for each feature
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Include the target variable in the correlation matrix
corr_matrix = data.corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


