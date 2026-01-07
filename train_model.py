import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('Gulab_heart_disease.xlsx')
data.head()
data.describe(include="all")
data.isnull().sum()

data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
data['BP'] = data['BP'].map({'High': 1, 'Normal': 0})
data['cholesterol'] = data['cholesterol'].map({'High': 1, 'Normal': 0})
data['smoking'] = data['smoking'].map({'Yes': 1, 'No': 0})

# Define features (X) and target (y) - Add 'sex' feature
X = data[['age', 'sex', 'BP', 'cholesterol', 'heart_rate', 'smoking']]
y = data['heart_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill NaN with median (ya mean)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

import pickle

# Save trained model
pickle.dump(model, open("heart_model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and Scaler saved successfully!")
