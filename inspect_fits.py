from astropy.io import fits

fits_path = "/Users/raghav/my files/Programing/grb-ml/data/AS1A04_207T01_9000002254cztM0_level1.fits"
hdul = fits.open(fits_path)

# Check all HDUs
for i, hdu in enumerate(hdul):
    print(f"HDU {i}: {hdu.name}")

# Now check 'CZT_QUAD1' for available columns
quad_data = hdul['CZT_QUAD1'].data
print("\n📦 Available columns in CZT_QUAD1:")
print(quad_data.columns.names)

hdul.close()
