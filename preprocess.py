import xarray as xr
import glob
import os

# 1. Find and sort your file paths
# Sorting is crucial to ensure time is monotonic (e.g., Jan, Feb, Mar)
file_paths = sorted(glob.glob("path/to/your/files/*.nc"))

datasets = []

# 2. Iterate, open, and load each file
for path in file_paths:
    # Using a context manager ('with') ensures the file handle closes after reading
    with xr.open_dataset(path) as ds:
        # .load() is key here: it forces data from disk into memory (NumPy)
        # immediately, so we can safely close the file handle.
        datasets.append(ds.load())

# 3. Concatenate the list of datasets along the time dimension
combined_ds = xr.concat(datasets, dim="time")

print(combined_ds)