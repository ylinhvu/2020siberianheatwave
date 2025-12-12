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
    'min_lat': 50, 'max_lat': 70,
    'min_lon': 65, 'max_lon': 120,
    # 'min_lat': 60, 'max_lat': 75, <--- from Overland paper
    # 'min_lon': 60, 'max_lon': 180, <--- from Overland paper
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

def select_region_and_time(ds, month_start=1, month_end=12, min_lat=None, max_lat=None, min_lon=None, max_lon=None, region=None, supress=False):
    """
    Selects a spatial region and a monthly time range from an xarray Dataset or DataArray.
    
    Args:
        ds (xr.Dataset or xr.DataArray): The input data with latitude, longitude, and time dimensions.
        month_start (int): The starting month (1-12).
        month_end (int): The ending month (1-12).
        min_lat, max_lat, min_lon, max_lon (float): User-defined lat/lon bounds.
        region (dict): A pre-defined region dictionary.
        supress (bool): Turn off/on system messages 

    Returns:
        xr.Dataset or xr.DataArray: The sliced data.
    """
    # Check for 'lat' and 'lon'. 'in dataset' checks both coords and data_vars
    if 'lat' in ds.coords:
        ds = ds.rename({"lat":"latitude"})
    if 'lon' in ds.coords:
        ds = ds.rename({"lon":"longitude"})
    
    # Check if a pre-defined region dictionary is provided
    if region is not None:
        if isinstance(region, dict):
            if not supress: print(f"Slicing data for the '{region.get('name', 'Unnamed')}' region.")
            is_increasing = np.all(np.diff(ds.latitude) > 0)
            
            ds_sliced = ds.sel(
                latitude=slice(region["min_lat"], region["max_lat"]) if is_increasing else slice(region["max_lat"], region["min_lat"]) , 
                longitude=slice(region["min_lon"], region["max_lon"])
            )
        else:
            raise TypeError("The 'region' argument must be a dictionary.")
    # Check if user has provided all four coordinate bounds
    elif all(v is not None for v in [min_lat, max_lat, min_lon, max_lon]):
        if not supress: print("Slicing data for user-defined region.")
        is_increasing = np.all(np.diff(ds.latitude) > 0)
        
        ds_sliced = ds.sel(
            latitude=slice(min_lat, max_lat) if is_increasing else slice(max_lat, min_lat), 
            longitude=slice(min_lon, max_lon)
        )
    else:
        if not supress: print("No specific region provided. Using the full spatial domain.")
        ds_sliced = ds

    # Make sure the common period between model and observations are selected (1979-2024)
    ds = ds.sel(time=slice("1979","2024"))
    
    # Filter by month(s)
    if not supress: print(f"Filtering for months: {month_start} through {month_end}.")
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

    # if 'time' not in da.dims or len(da.dims) > 1:
    #     print("ValueError(Input must be a 1D DataArray with a 'time' dimension.")
        
    # Store original attributes and coordinates
    original_coords = da.coords
    original_attrs = da.attrs
    
    nan_ind = np.isnan(da.values)
    da = xr.where(np.isnan(da.values), 1000, da.values) # fill nan with 1000

    da_detrend = signal.detrend(da, axis=0, type='linear') # return detrended data in ndarray

    da_detrend[nan_ind] = np.nan

    da_detrend = xr.DataArray(da_detrend, dims=['time', 'latitude', 'longitude'],
                            coords=original_coords,
                            attrs=original_attrs)
    
    # # Use the values for detrending, handling NaNs
    # valid_indices = ~np.isnan(da.values)
    # detrended_values = np.full_like(da.values, np.nan) # Create an array filled with NaNs
    
    # if np.any(valid_indices):
    #     detrended_values[valid_indices] = signal.detrend(da.values[valid_indices], axis=0, type='linear')
    # else:
    #     warnings.warn("DataArray contains only NaN values. Returning as is.")

    # # Reconstruct the DataArray
    # da_detrend = xr.DataArray(
    #     detrended_values,
    #     dims=['time'],
    #     coords=original_coords,
    #     attrs=original_attrs
    # )
    
    return da_detrend

def detrend_dataset(ds):
    """
    Detrends a dataset along the time dimension, handling NaNs.
    
    Args:
        ds (xr.Dataset): Input dataset with dimensions (time, lat, lon).
        
    Returns:
        xr.DataArray: The detrended data.
    """
    # 1. Handle Time Units
    # Polyfit works on numeric values. If time is datetime64, it converts to 
    # nanoseconds, making the slope very small. 
    # It is often better to convert time to "years" or "days" for interpretability.
    # Here we create a numeric time coordinate (e.g., fractional years).
    
    # Create a copy to avoid modifying the original index in place
    ds_calc = ds.copy()
    
    # 1. Identify grid cells that have at least one NaN along the time axis
    # .isnull().any(dim='time') results in a 2D mask (lat, lon)
    has_missing_data = ds_calc.isnull().any(dim='time')
    
    # 2. Apply the mask
    # .where() keeps values where the condition is True, and fills others with NaN.
    # We want to keep values where has_missing_data is False (~)
    ds_calc = ds_calc.where(~has_missing_data)

    # Convert datetime to a float (e.g., days since start)
    # This ensures the regression is calculated against a numeric X-axis
    ds_calc['time_numeric'] = (
        ds_calc['time'] - ds_calc['time'][0]
    ) / np.timedelta64(1, 'D')
    
    # Swap dims so we fit against the numeric time
    ds_calc = ds_calc.swap_dims({'time': 'time_numeric'})

    print("Calculating linear trend (this handles NaNs automatically)...")
    
    # 2. Fit the linear trend (degree=1)
    # skipna=True ensures that if a specific year is NaN, it is ignored 
    # in the fitting process for that specific grid cell.
    coeffs = ds_calc.polyfit(
        dim='time_numeric', 
        deg=1, 
        skipna=True
    )

    # 3. Evaluate the trend
    # polyval calculates the trend line (y = mx + b) using the coefficients
    trend = xr.polyval(ds_calc['time_numeric'], coeffs.polyfit_coefficients)

    # 4. Detrend
    # Subtract the calculated trend from the original data
    detrended = ds_calc - trend

    # 5. Cleanup
    # Swap dimensions back to original datetime if desired
    detrended = detrended.swap_dims({'time_numeric': 'time'})
    detrended = detrended.drop_vars('time_numeric')
    
    return detrended


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
    ds = ds[var]
    ds_regional = select_region_and_time(ds, **kwargs)
    
    # 2. Apply latitude weighting for accurate spatial averaging
    weights = np.cos(np.deg2rad(ds_regional.latitude))
    weights.name = "weights"

    # 3. Calculate anomalies
    gb = ds_regional.groupby('time.month')
    clim = gb.mean(dim='time')
    anom_full = gb - clim
    anom_full = anom_full.drop_vars('month')

    # 4. Spatially average the anomalies to get a single time series
    anom = anom_full.weighted(weights).mean(dim=('longitude', 'latitude'))

    # 5. Detrend the resulting time series
    anom_detrend = detrend_dataarray(anom)
    
    # 6. Standardize the anomalies
    # The anomaly is standardized by the standard deviation of that month's anomalies
    monthly_std = anom_detrend.groupby('time.month').std(dim='time')
    anom_detrend_standardized = anom_detrend.groupby('time.month') / monthly_std
    
    monthly_std = anom.groupby('time.month').std(dim='time')
    anom_standardized = anom.groupby('time.month') / monthly_std
    
    return anom, anom_standardized, anom_detrend, anom_detrend_standardized, clim

def preprocess_for_mca(ds, var, **kwargs):
    """
    Prepares data for MCA by calculating and detrending anomalies.
    This version keeps the spatial dimensions.

    Args:
        ds (xr.Dataset): The input dataset.
        var (str): The variable name to process.
        **kwargs: Arguments passed to select_region_and_time.

    Returns:
        xr.DataArray: A DataArray of detrended anomalies with lat/lon dimensions.
    """
     # 1. Select the region and time period of interest
    ds = ds[var]
    ds_regional = select_region_and_time(ds, **kwargs)
    
    # 2. Calculate anomalies
    gb = ds_regional.groupby('time.month')
    clim = gb.mean(dim='time')
    anom_full = gb - clim
    anom_full = anom_full.drop_vars('month')
    
    # 3. Detrend the anomalies while preserving spatial dimensions
    anom_detrend = detrend_dataset(anom_full)
    print(f"Finished preprocessing '{var}' for MCA.")
    return anom_detrend

def perform_mca(x, y, n_modes=12):
    """
    Performs Maximum Covariance Analysis (MCA) using the xeofs library.

    Args:
        x (xr.DataArray): The left field for analysis (time, lat, lon).
        y (xr.DataArray): The right field for analysis (time, lat, lon).
        n_modes (int): The number of modes to compute.

    Returns:
        xeofs.cross.MCA: The fitted MCA model object.
    """
    print(f"Performing MCA for {n_modes} modes...")
    # Initialize the MCA model
    # use_coslat=True applies latitude correction automatically
    mca = xe.cross.MCA(n_modes=n_modes, standardize=False, use_coslat=True)
    
    # Fit the model to the two data fields
    mca.fit(x, y, dim="time")
    
    print("MCA fitting complete.")
    return mca

def get_significant_pattern_masks(mca_model, p_value=0.05):
    """
    Creates boolean masks for statistically significant areas in MCA patterns.

    Args:
        mca_model (xe.cross.MCA): A fitted MCA model from perform_mca.
        p_value (float): The significance level (e.g., 0.05 for 95% confidence).

    Returns:
        tuple: A tuple containing:
            - hom_mask (list of xr.DataArray): Masks for homogeneous patterns.
            - het_mask (list of xr.DataArray): Masks for heterogeneous patterns.
    """
    # Get p-values for both homogeneous and heterogeneous patterns
    _, pvals_hom = mca_model.homogeneous_patterns()
    _, pvals_het = mca_model.heterogeneous_patterns()
    
    # Create a boolean mask where the p-value is less than the threshold
    hom_mask = [p < p_value for p in pvals_hom]
    het_mask = [p < p_value for p in pvals_het]
    
    return hom_mask, het_mask