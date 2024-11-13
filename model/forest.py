import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load and Preprocess the Data
data = pd.read_csv("ecological_health_dataset.csv")
data = data.drop(columns=['Timestamp'])
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())
valid_labels = [1, 2, 3, 4]
data = data[data['Ecological_Health_Label'].isin(valid_labels)]
data = data.reset_index(drop=True)

# Adjust labels to start from 0 for compatibility
X = data.drop(columns=['Ecological_Health_Label']).values
y = data['Ecological_Health_Label'].astype(int).values - 1

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 2: Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Compute Class Weights (optional for Random Forest but helps with imbalance)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Step 4: Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
rf.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Step 6: Feature Importance Analysis
importances = rf.feature_importances_
feature_names = data.columns[:-1]
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

# Optional: Selecting Top Features for further refinement or other analysis
N = 8
top_features = feature_importance_df['Feature'].head(N).tolist()
print("\nTop Features:\n", top_features)
