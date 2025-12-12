import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import xarray as xr
import calendar

from cartopy.crs import Orthographic, PlateCarree, Robinson
from cartopy.feature import LAND
from matplotlib.gridspec import GridSpec

def plot_mca_loadings(scores, modes=3, save=False):
    fig, ax = plt.subplots(modes,2,figsize=(10, 8))

    for i in range(modes):
        ax[i,0].axhline(y=0, color='grey', linestyle='--',linewidth=0.9)
        ax[i,1].axhline(y=0, color='grey', linestyle='--',linewidth=0.9)
        
        normalized_pc1=scores[0].sel(mode=i+1)
        normalized_pc2=scores[1].sel(mode=i+1)
        ax[i,0].plot(scores[0].sel(mode=i+1).time,normalized_pc1,marker='o')
        ax[i,1].plot(scores[1].sel(mode=i+1).time,normalized_pc2,marker='o')

        ax[i,0].set_xlim(scores[0].time.min(),scores[0].time.max())
        ax[i,1].set_xlim(scores[0].time.min(),scores[0].time.max())
        years = mdates.YearLocator(5)  # Every 10 years
        ax[i,0].xaxis.set_major_locator(years)
        ax[i,1].xaxis.set_major_locator(years)
        ax[i,0].set_xlabel("")
        ax[i,1].set_xlabel("")
        ax[i,0].set_ylabel("Mode "+str(i+1)+"\nSST")
        ax[i,1].set_ylabel("2m Temperature")
        ax[i,0].set_ylim(-0.5,0.5)
        ax[i,1].set_ylim(-0.5,0.5)
        ax[i,0].grid()
        ax[i,1].grid()
        # ax[i,0].set_title("")
        # ax[i,1].set_title("")
    plt.suptitle("JFM Normalized PC Scores for Modes 1-",modes)
    plt.tight_layout()
    
    if save == True:
        plt.savefig(save)
    return

def plot_mca_pc_scores():
    return

def plot_climatology(p1, p2=None, ax=None, standardized=False, save=False):
    
    # Plotting the first line (With Trend)
    p1.plot(
        ax=ax,
        color='olivedrab',
        marker='o',
        linestyle='-',
        linewidth=2.5, # Slightly thicker line
        markersize=8,  # Larger markers
    )
    
    if p2 is not None:
    # Plotting the second line (Without Trend)
        p2.plot(
            ax=ax,
            color='mediumorchid',
            marker='X',
            linestyle='--',
            linewidth=2.5,
            markersize=8,
        )
    
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    time_range = range(int(p1.time.dt.month[0]), int(p1.time.dt.month[-1]+1))
    month_labels = [calendar.month_abbr[i] for i in time_range] # Set custom x-axis ticks to show month names
    ax.set_xticks(ticks=time_range)
    ax.set_xticklabels(month_labels)                       
    ax.set_xlabel("Month", fontsize=12)
    ax.set_xlim(0.5, 12.5) # Add a little padding to the x-axis

    if standardized==True:
        ax.set_ylim(-4, 4)
        plt.ylabel("Standardized [σ]")
    else:
        plt.ylabel(f"{p1.units}")
    
    if p2 is not None:
        ax.legend(fontsize=11, loc='best')

def plot_2020_anomaly(p1, p2, ax=None, standardized=False, save=False):
    
    # Plotting the first line (With Trend)
    ax.plot(
        p1.time.dt.month, p1,
        color='olivedrab',
        marker='o',
        linestyle='-',
        linewidth=2.5, # Slightly thicker line
        markersize=8,  # Larger markers
        label="With Linear Trend"
    )
    
    # Plotting the second line (Without Trend)
    ax.plot(
        p2.time.dt.month, p2,
        color='mediumorchid',
        marker='X',
        linestyle='--',
        linewidth=2.5,
        markersize=8,
        label="Without Linear Trend"
    )
    
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    time_range = range(int(p1.time.dt.month[0]), int(p1.time.dt.month[-1]+1))
    month_labels = [calendar.month_abbr[i] for i in time_range] # Set custom x-axis ticks to show month names
    ax.set_xticks(ticks=time_range)
    ax.set_xticklabels(month_labels)                       
    ax.set_xlabel("Month", fontsize=12)
    ax.set_xlim(0.5, 12.5) # Add a little padding to the x-axis

    if standardized==True:
        ax.set_ylim(-4, 4)
        plt.ylabel("Standardized Anomaly [σ]")
    else:
        plt.ylabel(f"Anomaly [{p1.units}]")
        
    ax.legend(fontsize=11, loc='best')