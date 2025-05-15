import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs #type: ignore
import cartopy.feature as cfeature #type: ignore
import numpy as np
import pymannkendall as mk
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter

# Path to the new dataset file
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp-COMPLETE.grib'

# Reading the file
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting necessary data
prcp = ds['tp']*1000  # Convert from m to mm
lat = ds['latitude']
lon = ds['longitude']

# Start date of the dataset
start_date = np.datetime64('1950-01-31T00:00:00')
prcp = prcp.sel(time=slice(start_date, None))

# Resampling to an annual frequency
prcp = prcp.resample(time='1Y').sum()

# Initialize an array to store the Sen's slope results
slope_results = np.empty((prcp.shape[1], prcp.shape[2]))

# Loop through each grid point to calculate the Sen's slope
for i in range(prcp.shape[1]):
    for j in range(prcp.shape[2]):
        # Extract the time series for the grid point
        time_series = prcp[:, i, j].values
        
        # Calculate the Mann-Kendall trend
        trend = mk.original_test(time_series, alpha=0.05)
        
        # Store the Sen's slope (trend.slope) in the results array
        slope_results[i, j] = trend.slope if trend.slope is not None else np.nan


# Create a spatial trend color map using the slope_results
fig, ax = plt.subplots(figsize=(20, 18), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=900)
ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

# Adding colorbar with normalization to center zero
vmin = np.nanmin(slope_results)
vmax = np.nanmax(slope_results)
vcenter = 0

# Multiply vmin and vmax by 10 to convert to mm/decade
vmin *= 10
vmax *= 10

# Ensure vmin, vcenter, and vmax are in ascending order
if vmin >= vcenter * 10:
    vmin = vcenter * 10 - 1  # Adjust vmin to be less than vcenter
if vmax <= vcenter * 10:
    vmax = vcenter * 10 + 1  # Adjust vmax to be greater than vcenter

norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter * 10, vmax=vmax)

# Multiply slope_results by 10 for plotting in mm/decade
slope_results_decade = slope_results * 10

# Plot the trend results
contour = ax.contourf(lon, lat, slope_results_decade, transform=ccrs.PlateCarree(), cmap='viridis_r', norm=norm, extend='both')
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
cbar.set_label("Rainfall Increase (mm/decade)", fontsize=22)
cbar.ax.tick_params(labelsize=20)

# Add title
ax.set_title("Rainfall Trends in Region 8", fontsize=30)

plt.show()


### TO REFINE THE CODE

# # Extracting the p-value from the Mann-Kendall test
# # Initialize an array to store the p-values
# p_values = np.empty((prcp.shape[1], prcp.shape[2]))

# # Loop through each grid point to calculate the Sen's slope
# for i in range(prcp.shape[1]):
#     for j in range(prcp.shape[2]):

#         # Store the P-values (trend.p) in the p_values array
#         p_values[i, j] = trend.p if trend.p is not None else np.nan

# # Creating a spatial trend map using the extracted p-values
# fig, ax = plt.subplots(figsize=(20, 18), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=900)
# ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

# # Adding colorbar with normalization to center zero
# vmin = np.nanmin(p_values)
# vmax = np.nanmax(p_values)
# vcenter = 0

# if vmin >= vcenter * 10:
#     vmin = vcenter * 10 - 1  # Adjust vmin to be less than vcenter
# if vmax <= vcenter * 10:
#     vmax = vcenter * 10 + 1  # Adjust vmax to be greater than vcenter

# norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter * 10, vmax=vmax)

# # Plot the trend results
# contour = ax.contourf(lon, lat, p_values, transform=ccrs.PlateCarree(), cmap='viridis', norm=norm, extend='both')
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')

# # Add gridlines for latitude and longitude
# gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.7)
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 15}
# gl.ylabel_style = {'size': 15}

# # Adding a single colorbar for the plot
# cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, aspect=50, shrink=0.85)

# class ScientificFormatter(ScalarFormatter):  # Scientific notation converter
#     def _set_format(self):
#         self.format = "%1.1e"

#     def _set_order_of_magnitude(self):
#         # Force the order of magnitude to be -5
#         self.orderOfMagnitude = -5

# cbar.formatter = ScientificFormatter()
# cbar.update_ticks()

# cbar.set_label("P-values", fontsize=22)
# cbar.ax.tick_params(labelsize=20)

# # Add title
# ax.set_title("Trend Significance Spatial Map", fontsize=30)

# plt.show()