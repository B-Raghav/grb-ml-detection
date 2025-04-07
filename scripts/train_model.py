import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), '../data/grb_dataset.csv')
df = pd.read_csv(csv_path)

# Create additional features
df['prev'] = df['counts'].shift(1).fillna(0)
df['next'] = df['counts'].shift(-1).fillna(0)

# Features and labels
X = df[['counts', 'prev', 'next']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = os.path.join(os.path.dirname(__file__), '../models/grb_rf_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")
