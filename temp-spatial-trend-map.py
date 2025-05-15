import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs #type: ignore
import cartopy.feature as cfeature #type: ignore
import numpy as np
import pymannkendall as mk
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter


# Path to the new dataset file
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp.grib'

# Reading the file
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting necessary data
temp = ds['t2m']
lat = ds['latitude']
lon = ds['longitude']

# Convert temperature from Kelvin to Celsius
temp_celsius = temp - 273.15

# Resampling to an annual frequency
temp_celsius = temp_celsius.resample(time='1Y').mean()

# Filter the data for the period 1961-1990 (baseline period)
baseline_period = temp_celsius.sel(time=slice('1961-01-01', '1990-12-01'))

# Calculate the average temperature over the baseline period
baseline_temp_ave = baseline_period.mean(dim='time')

# Calculate the anomalies per grid point
temp_anomalies = temp_celsius - baseline_temp_ave

# Initialize an array to store the Sen's slope results
slope_results = np.empty((temp_anomalies.shape[1], temp_anomalies.shape[2]))

# Loop through each grid point to calculate the Sen's slope
for i in range(temp_anomalies.shape[1]):
    for j in range(temp_anomalies.shape[2]):
        # Extract the time series for the grid point
        time_series = temp_anomalies[:, i, j].values
        
        # Calculate the Mann-Kendall trend
        trend = mk.original_test(time_series, alpha=0.05)
        
        # Store the Sen's slope (trend.slope) in the results array
        slope_results[i, j] = trend.slope if trend.slope is not None else np.nan

# Multiply slope_results by 10 to convert to °C/decade
slope_results_decade = slope_results*10

# Create a spatial trend color map using the slope_results
fig, ax = plt.subplots(figsize=(20, 18), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=900)
ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

# Adding colorbar with normalization to center zero
vmin = 0.06
vmax = 0.24
vcenter = (vmin + vmax) / 2  

norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# Multiply slope_results by 10 for plotting in °C/decade
slope_results_decade = slope_results * 10

# Plot the trend results using a red-dark red colormap
contour = ax.contourf(lon, lat, slope_results_decade, transform=ccrs.PlateCarree(), cmap='Reds', extend='both', norm=norm)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Add a legend-like title within the plot at the upper-right
ax.text(0.95, 0.95, 'p-value < 0.05', transform=ax.transAxes, fontsize=20, ha='right', va='top', 
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

# Add gridlines for latitude and longitude
gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# Adding a single colorbar for the plot
cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=0.85)
cbar.set_label("Temperature Increase (°C/decade)", fontsize=22)
cbar.ax.tick_params(labelsize=20, direction='in', which='both')

# Add title
ax.set_title("Temperature Trends in Region 8", fontsize=30)

plt.show()