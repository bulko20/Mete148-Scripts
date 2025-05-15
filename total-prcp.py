# Importing necessary libraries
import xarray as xr
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy.crs as ccrs #type: ignore
import cartopy.feature as cfeature #type: ignore
import pymannkendall as mk
from matplotlib.lines import Line2D

# Path to dataset
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp-COMPLETE.grib'

# Opening the dataset
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting the dataset and converting to mm
tp_mm = ds['tp']*1000

# Start date of the dataset
start_date = np.datetime64('1950-01-31T00:00:00')
tp_mm = tp_mm.sel(time=slice(start_date, None))

# Calculating the average of total precipitation over all years
tp_mm_avg = tp_mm.mean(dim='time')

### UNCOMMENT THE CODE BELOW TO PLOT THE AVERAGE TOTAL PRECIPITATION FOR ALL YEARS ###

# # Plotting the average total precipitation for all years
# fig = plt.figure(figsize=(20, 10), dpi=1200)
# ax = plt.axes(projection=ccrs.PlateCarree())

# # Create the filled contour plot
# contour = ax.contourf(tp_mm_avg['longitude'], tp_mm_avg['latitude'], tp_mm_avg, 
#                       transform=ccrs.PlateCarree(), cmap='Blues', extend='both')

# # Adding coastlines and borders
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')

# # Adding gridlines and labels
# gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 12}
# gl.ylabel_style = {'size': 12}

# # Adding colorbar
# cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
# cbar.set_label('Precipitation (mm)')

# # Adding title
# plt.title('Average Total Precipitation (1950-2024)', fontsize=20)

# # Show the plot
# plt.show()

### END OF AVERAGE TOTAL PRECIPITATION PLOT ###

# Creating subplots for total precipitation in 1950, 1975, 2000, and 2024
years_to_plot = [1950, 1975, 2000, 2024]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=1200)

for ax, year in zip(axes.flat, years_to_plot):
    tp_year = tp_mm.sel(time=str(year))
    tp_year_2d = tp_year.isel(time=0)
    contour = ax.contourf(tp_year_2d['longitude'], tp_year_2d['latitude'], tp_year_2d, 
                          transform=ccrs.PlateCarree(), cmap='Blues', extend='both')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Adding gridlines and labels
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    # Add a legend-like title within the subplot at the upper-right
    ax.text(0.95, 0.95, f'{year}', transform=ax.transAxes, fontsize=15, ha='right', va='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

# Adjust layout to reduce spaces between plots
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Adding a single colorbar for all subplots
cbar = fig.colorbar(contour, ax=axes, orientation='vertical', pad=0.03, aspect=40, shrink=0.85)
cbar.set_label('Precipitation (mm)', fontsize=25)
cbar.ax.tick_params(labelsize=20)

plt.suptitle('Total Precipitation for Selected Years', fontsize=30, y=0.905)
plt.show()


### ALTERNATIVE PLOT: Uncomment the code below to plot the time series of average total precipitation ###

# Plotting the average total precipitation using a map
# fig = plt.figure(figsize=(10, 5))
# ax = plt.axes(projection=ccrs.PlateCarree())
# tp_mm_avg.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', cbar_kwargs={'label': 'Precipitation (mm)'})
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')

# Adding gridlines and labels
# gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 12}
# gl.ylabel_style = {'size': 12}

# Adding title
# plt.title('Average Total Precipitation (1950-2024)')
# plt.show()

### END OF ALTERNATIVE PLOT ###

# Resampling the dataset into yearly
tp_mm_annual = tp_mm.resample(time='Y').mean()

# Computing tp_mm_annual longitudinal and latitudinal average to get 1D data
tp_1d = tp_mm_annual.mean(dim=['latitude', 'longitude'])

# Convert time values to numerical format (e.g., number of years since the first year)
years = tp_1d['time'].dt.year.values

# Testing for trend significance
result = mk.original_test(tp_1d)

# Solving for the trend line
years_numeric = np.array([year.astype('datetime64[Y]').astype(int) + 1970 for year in years])
slope, intercept, r_value, p_value, std_err = linregress(years_numeric, tp_1d)

# Plotting the time series of average total precipitation
plt.figure(figsize=(20, 10))
plt.plot(years, tp_1d, label='Observed Precipitation')
plt.plot(years, slope * years_numeric + intercept, label='Trend Line', color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Precipitation (mm)', fontsize=15)
plt.title('Annual Mean Total Precipitation in Region 8 (1950-2024)', fontsize=20)

# Create custom legend entry for the MK test results
custom_legend = Line2D([0], [0], color='white', label=f'P-value: {result.p:.3f}\nMK Statistic: {result.z:.3f}')

# Adjust the legend position to avoid overlap
plt.legend(loc='upper left', fontsize=15, handles=[plt.Line2D([0], [0], color='blue', label='Observed Precipitation', linewidth=1.5),
                                                   plt.Line2D([0], [0], color='red', label='Trend Line (Slope: 0.0303)', linestyle='--', linewidth=3),
                                                   custom_legend])

plt.grid()
plt.show()

# Printing the results to check for trend
print(f"MK Test Results:")
print(f"  Trend: {result.trend}")
print(f"  H: {result.h}")
print(f"  P-value: {result.p:.3f}")
print(f"  Z: {result.z}")
print(f"  Tau: {result.Tau}")
print(f"  S: {result.s}")
print(f"  Var(S): {result.var_s}")
print(f"  Slope: {result.slope}")

# # COLORMAP FOR PRECIPITATION LOOPING THROUGH ALL YEARS (see behavior of precipitation throughout the years)
# # Create a directory to save the plots
# output_dir = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\2nd Sem\Research 2\Plots\prcp-plots\output_dir'
# os.makedirs(output_dir, exist_ok=True)

# # Loop over each year and create a plot for the average total precipitation for that year
# for year in range(1950, 2025):

#     # Select the precipitation for the current year
#     prcp_for_year = tp_mm.sel(time=str(year)).mean(dim='time')
    
#     # Plotting the average total precipitation for all years
#     fig = plt.figure(figsize=(10, 10))  
#     ax = plt.axes(projection=ccrs.PlateCarree())

#     # Create the filled contour plot
#     contour = ax.contourf(prcp_for_year['longitude'], prcp_for_year['latitude'], prcp_for_year, 
#                         transform=ccrs.PlateCarree(), cmap='Blues', extend='both')

#     # Adding coastlines and borders
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')

#     # Adding gridlines and labels
#     gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False
#     gl.xlabel_style = {'size': 12}
#     gl.ylabel_style = {'size': 12}

#     # Adding colorbar
#     cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, shrink=0.5)
#     cbar.set_label('Precipitation (mm)')

#     # Adding title
#     plt.title(f'Precipitation for {year}', fontsize=20)

#     # Save the plot to a file
#     plt.savefig(os.path.join(output_dir, f'precipitation_{year}.png'))
#     plt.close(fig)

# print(f"Plots saved to {output_dir}")

### END OF CODE ###
