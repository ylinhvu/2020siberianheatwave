import xarray as xr
import numpy as np
from scipy import signal

def standardized_anom(ds,var,month_start=1,month_end=12,min_lat=None,max_lat=None,min_lon=None,max_lon=None,region=None):
    """
    Calculates the detrended, standardized anomaly for a given time series.

    The process is:
    1. Calculate monthly climatology over a base period.
    2. Calculate anomaly by subtracting the climatology.
    3. Detrend the resulting anomaly.
    4. Standardize the detrended anomaly by its standard deviation.
        
    Input:
    ds = xarray dataset of climate data
    var = variable string
    min_lat, max_lat = choose the latitude bounds for regional analysis
    min_lon, max_lon = choose the longitude bounds for regional analysis
    month_start, month_end = choose the month start and end for seasonal analysis

    Output:
    var_anom = anomaly dataset
    """
    # --- 1. Calculate Climatology ---
    # Select the base period and region for calculating the long-term average
    if region is not None:
        print("Pre-processing "+str(region))
        ds = ds.sel(latitude=slice(region["max_lat"],region["min_lat"]), longitude=slice(region["min_lon"],region["max_lon"]))
        ds = ds.isel(time=ds.time.dt.month.isin(np.arange(month_start,month_end+1)))
    elif (min_lat,max_lat,min_lon,max_lon) is not None:
        print("Pre-processing user-defined region")
        ds = ds.sel(latitude=slice(max_lat,min_lat), longitude=slice(min_lon,max_lon)) # change this order depending on dataset
        ds = ds.isel(time=ds.time.dt.month.isin(np.arange(month_start,month_end+1)))
    else:
        print("Pre-processing all latitudes and longitudes")

    ds = ds[var]
    climatology = ds.groupby('time.month').mean('time')
    
    # --- 2. Calculate Anomaly ---
    # Subtract the climatology from the full time series
    anomaly = ds.groupby('time.month') - climatology
    
    # --- 3. Detrend the Anomaly ---
    # Use the existing detrending logic on the 'Anomaly' variable
    not_nan_mask = ~np.isnan(anomaly)
    detrended_values = signal.detrend(anomaly.values[not_nan_mask.values])
    
    detrended_anomaly = xr.full_like(anomaly, fill_value=np.nan)
    detrended_anomaly.values[not_nan_mask.values] = detrended_values
    
    # --- 4. Calculate Standard Deviation of the Detrended Anomaly ---
    std_dev = detrended_anomaly.std(dim='time')

    # --- 5. Standardize the Anomaly ---
    standardized_anomaly = detrended_anomaly / std_dev
    
    return climatology, anomaly, standardized_anomaly