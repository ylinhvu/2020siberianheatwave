import numpy as np
import xarray as xr
import pandas as pd
from scipy import signal
import xeofs as xe
import warnings

# --- Region Definitions ---
# Pre-defined dictionaries for common geographical regions.
siberia = {
    'name': 'Siberia',
    'min_lat': 60, 'max_lat': 75,
    'min_lon': 60, 'max_lon': 180,
}
eurasia = {
    'name': 'Eurasia',
    'min_lat': 40, 'max_lat': 80,
    'min_lon': 0, 'max_lon': 180,
}

arctic = {
    'name': 'Arctic',
    'min_lat': 66.5, 'max_lat': 90,
    'min_lon': 0, 'max_lon': 360,
}

north_atlantic = {
    'name': 'North Atlantic',
    'min_lat': 0, 'max_lat': 90,
    'min_lon': 275, 'max_lon': 360,
}

# --- Core Functions ---

def select_region_and_time(ds, month_start=1, month_end=12, min_lat=None, max_lat=None, min_lon=None, max_lon=None, region=None):
    """
    Selects a spatial region and a monthly time range from an xarray Dataset or DataArray.
    
    Args:
        ds (xr.Dataset or xr.DataArray): The input data with latitude, longitude, and time dimensions.
        month_start (int): The starting month (1-12).
        month_end (int): The ending month (1-12).
        min_lat, max_lat, min_lon, max_lon (float): User-defined lat/lon bounds.
        region (dict): A pre-defined region dictionary.

    Returns:
        xr.Dataset or xr.DataArray: The sliced data.
    """
    # Check if a pre-defined region dictionary is provided
    if region is not None:
        if isinstance(region, dict):
            print(f"Slicing data for the '{region.get('name', 'Unnamed')}' region.")
            ds_sliced = ds.sel(
                latitude=slice(region["max_lat"], region["min_lat"]), 
                longitude=slice(region["min_lon"], region["max_lon"])
            )
        else:
            raise TypeError("The 'region' argument must be a dictionary.")
    # Check if user has provided all four coordinate bounds
    elif all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
        print("Slicing data for user-defined region.")
        ds_sliced = ds.sel(
            latitude=slice(max_lat, min_lat), 
            longitude=slice(min_lon, max_lon)
        )
    else:
        print("No specific region provided. Using the full spatial domain.")
        ds_sliced = ds

    # Filter by month(s)
    print(f"Filtering for months: {month_start} through {month_end}.")
    # Handle cases where the year wraps around (e.g., December-January-February)
    if month_start > month_end:
        months = list(range(month_start, 13)) + list(range(1, month_end + 1))
        ds_sliced = ds_sliced.isel(time=ds_sliced.time.dt.month.isin(months))
    else:
        months = list(range(month_start, month_end + 1))
        ds_sliced = ds_sliced.isel(time=ds_sliced.time.dt.month.isin(months))
        
    return ds_sliced

def detrend_dataarray(da):
    """
    Removes the linear trend from a 1D xarray DataArray.
    Handles NaN values gracefully.

    Args:
        da (xr.DataArray): A 1D DataArray with a 'time' dimension.

    Returns:
        xr.DataArray: The detrended DataArray.
    """
    if 'time' not in da.dims or len(da.dims) > 1:
        raise ValueError("Input must be a 1D DataArray with a 'time' dimension.")
        
    # Store original attributes and coordinates
    original_coords = da.coords
    original_attrs = da.attrs
    
    # Use the values for detrending, handling NaNs
    valid_indices = ~np.isnan(da.values)
    detrended_values = np.full_like(da.values, np.nan) # Create an array filled with NaNs
    
    if np.any(valid_indices):
        detrended_values[valid_indices] = signal.detrend(da.values[valid_indices], axis=0, type='linear')
    else:
        warnings.warn("DataArray contains only NaN values. Returning as is.")

    # Reconstruct the DataArray
    da_detrend = xr.DataArray(
        detrended_values,
        dims=['time'],
        coords=original_coords,
        attrs=original_attrs
    )
    
    return da_detrend

def calculate_detrended_stats(ds, var, **kwargs):
    """
    Calculates climatology, anomalies, and detrended/standardized anomalies.

    Args:
        ds (xr.Dataset): The input dataset.
        var (str): The variable name to process (e.g., 't2m').
        **kwargs: Arguments to be passed to select_region_and_time.

    Returns:
        tuple: A tuple containing:
            - clim (xr.DataArray): The monthly climatology.
            - anom (xr.DataArray): The spatially averaged anomalies.
            - anom_detrend (xr.DataArray): The detrended spatially averaged anomalies.
            - anom_detrend_std (xr.DataArray): The standardized detrended anomalies.
    """
    # 1. Select the region and time period of interest
    ds_regional = select_region_and_time(ds, **kwargs)
    
    # 2. Apply latitude weighting for accurate spatial averaging
    weights = np.cos(np.deg2rad(ds_regional.latitude))
    weights.name = "weights"

    # 3. Select the variable and calculate anomalies
    data_array = ds_regional[var]
    gb = data_array.groupby('time.month')
    clim = gb.mean(dim='time')
    anom_full = gb - clim

    # 4. Spatially average the anomalies to get a single time series
    anom_spatial_avg = anom_full.weighted(weights).mean(dim=('longitude', 'latitude'))

    # 5. Detrend the resulting time series
    anom_detrend = detrend_dataarray(anom_spatial_avg)
    
    # 6. Standardize the detrended anomalies
    # The anomaly is standardized by the standard deviation of that month's anomalies
    monthly_std = anom_detrend.groupby('time.month').std(dim='time')
    anom_detrend_standardized = anom_detrend.groupby('time.month') / monthly_std
    
    return clim, anom_spatial_avg, anom_detrend, anom_detrend_standardized

def calcClimAnom_model(ds,min_lat,max_lat,min_lon,max_lon,month_start,month_end):
    if np.all(np.diff(ds.lat) > 0)==True: # check if latitude is monotonically increasing
        ds_sub = ds.tas.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon)) # change order of lat slice
        ds_sub = ds_sub.isel(time=ds_sub.time.dt.month.isin(np.arange(month_start,month_end+1)))
    else: # then the lat array is flipped and monotonically decreasing
        ds_sub = ds.tas.sel(lat=slice(max_lat,min_lat), lon=slice(min_lon,max_lon)) # change order of lat slice
        ds_sub = ds_sub.isel(time=ds_sub.time.dt.month.isin(np.arange(month_start,month_end+1)))

    weights = np.cos(np.deg2rad(ds_sub.lat))
    weights.name = "weights"

    ds_sub=detrend(ds_sub)

    clim=ds_sub.weighted(weights).mean(dim=('longitude', 'latitude'))
    anom=clim
    clim=ds_sub.groupby('time.month').mean(dim='time') # generate monthly climatology
    anom=anom.groupby('time.month') - clim # generate monthly anomaly
   
    avg_anom = anom.sel(time=anom.time.dt.year==2020).mean() # average over all months

    assert not np.any(np.isnan(avg_anom)) # if nan is calculated, stop the function
    
    return avg_anom.values

def calcClimAnom_obs(ds,min_lat,max_lat,min_lon,max_lon,month_start,month_end):
    if np.all(np.diff(ds.latitude) > 0)==True: # check if latitude is monotonically increasing
        ds_sub = ds.sel(latitude=slice(min_lat,max_lat), longitude=slice(min_lon,max_lon)) # change order of lat slice
        ds_sub = ds_sub.isel(time=ds_sub.time.dt.month.isin(np.arange(month_start,month_end+1)))
    else: # then the lat array is flipped and monotonically decreasing
        ds_sub = ds.sel(latitude=slice(max_lat,min_lat), longitude=slice(min_lon,max_lon))
        ds_sub = ds_sub.isel(time=ds_sub.time.dt.month.isin(np.arange(month_start,month_end+1)))
    
    weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"

    # detrend after the region is selected
    ds_sub = detrend(ds_sub)

    # average over area before calculating clim and anom
    clim=ds_sub.weighted(weights).mean(dim=('longitude', 'latitude'))
    anom=clim
    clim=clim.groupby('time.month').mean(dim='time') # generate monthly climatology
    anom=anom.groupby('time.month') - clim # generate monthly anomaly

    avg_anom = anom.sel(time=anom.time.dt.year==2020).mean() # average over all months

    assert not np.any(np.isnan(avg_anom)) # if nan is calculated, stop the function
    
    return avg_anom.values