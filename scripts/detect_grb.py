import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import joblib

# Load FITS file
fits_path = os.path.join(os.path.dirname(__file__), '../data/AS1A04_207T01_9000002254cztM0_level1.fits')
hdul = fits.open(fits_path)
data = hdul['CZT_QUAD1'].data

# Extract time and count features
time = np.array(data['Time'])
data_array = data['DataArray']

# Compute count-based features for each time step
counts = np.array([np.count_nonzero(row) for row in data_array])
mean_diff = np.array([np.mean(np.diff(row[row != 0])) if np.count_nonzero(row) > 1 else 0 for row in data_array])
stddev = np.array([np.std(row[row != 0]) if np.count_nonzero(row) > 0 else 0 for row in data_array])

# Stack into feature matrix
features = np.vstack([counts, mean_diff, stddev]).T  # shape: (samples, 3)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), '../models/grb_rf_model.pkl')
model = joblib.load(model_path)

# Predict
predictions = model.predict(features)

# Filter detections
grb_times = time[predictions == 1]
grb_counts = counts[predictions == 1]

# Plot
# Downsample for better visualization (optional, adjust as needed)
step = 10
time_ds = time[::step]
counts_ds = counts[::step]

# Plot
plt.figure(figsize=(14, 6))
plt.plot(time_ds, counts_ds, label='Counts', alpha=0.7, linewidth=0.8)
plt.scatter(grb_times, grb_counts, color='red', label='Detected GRB', s=30, zorder=5)

plt.xlabel('Time')
plt.ylabel('Counts')
plt.title('GRB Detection Over Time')
plt.legend()
plt.grid(True)

plot_path = os.path.join(os.path.dirname(__file__), '../data/detection_plot_cleaned.png')
plt.savefig(plot_path, dpi=300)
print(f"✅ Cleaner detection plot saved to {plot_path}")