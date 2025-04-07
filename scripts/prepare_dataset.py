import os
import numpy as np
import pandas as pd
from astropy.io import fits

# Load the file
fits_path = os.path.join(os.path.dirname(__file__), '../data/AS1A04_207T01_9000002254cztM0_level1.fits')
hdul = fits.open(fits_path)

# List of QUADs to process
quad_names = ["CZT_QUAD1", "CZT_QUAD2", "CZT_QUAD3", "CZT_QUAD4"]

all_times = []

# Collect TIME data from each QUAD
for name in quad_names:
    data = hdul[name].data
    times = data['TIME']
    all_times.extend(times)

hdul.close()

# Convert to numpy array and sort
all_times = np.array(all_times)
all_times.sort()

# Bin size in seconds
bin_size = 1.0

# Create histogram
min_time = all_times.min()
max_time = all_times.max()
num_bins = int((max_time - min_time) / bin_size)
hist, bin_edges = np.histogram(all_times, bins=num_bins, range=(min_time, max_time))

# Make a DataFrame
df = pd.DataFrame({
    'time_bin': bin_edges[:-1],
    'counts': hist,
    'label': np.zeros_like(hist)  # Placeholder labels (0 = no GRB, 1 = GRB)
})

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), '../data/grb_dataset.csv')
df.to_csv(output_path, index=False)

print(f"✅ Dataset saved to {output_path}")
