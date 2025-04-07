import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Generate dummy burst-like data
np.random.seed(42)
counts = np.random.poisson(5, 1000).astype(float)

# Simulate some GRBs (spikes)
burst_indices = np.random.choice(range(100, 900), size=10, replace=False)
counts[burst_indices] += np.random.randint(20, 50, size=10)

# Create features (counts, prev, next)
prev = np.roll(counts, 1)
next_ = np.roll(counts, -1)
prev[0] = counts[0]
next_[-1] = counts[-1]

X = np.vstack((counts, prev, next_)).T

# Simulate labels (1 = burst, 0 = no burst)
y = np.zeros_like(counts)
y[burst_indices] = 1

# Train simple model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Predict
predictions = model.predict(X)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(counts, label="Counts")
plt.scatter(np.where(predictions == 1), counts[predictions == 1], color='red', label="Predicted GRB", zorder=5)
plt.xlabel("Time bin")
plt.ylabel("Counts")
plt.title("Simulated GRB Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
