import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs #type: ignore
import cartopy.feature as cfeature #type: ignore
from matplotlib.lines import Line2D
import numpy as np



# Path to the new dataset file
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp.grib'

# Reading the file
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting temperature data
temp = ds['t2m']

# Convert temperature from Kelvin to Celsius
temp_celsius = temp - 273.15

# Filter the data for the period 1961-1990 (baseline period)
baseline_period = temp_celsius.sel(time=slice('1961-01-01', '1990-12-01'))

# Calculate the average temperature over the baseline period
baseline_temp_ave = baseline_period.mean(dim='time')

# Display the average temperature data
print(baseline_temp_ave)

# Plotting the temperature anomalies using contourf
fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Create the filled contour plot
contour = ax.contourf(baseline_temp_ave['longitude'], baseline_temp_ave['latitude'], baseline_temp_ave, 
                      transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both')

# Adding coastlines and borders
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Adding gridlines and labels
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# Adding colorbar
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
cbar.set_label('Temperature (°C)')

# Adding title
plt.title('Average Temperature (1961-1990)', fontsize=20)

# Show the plot
plt.show()


### ALTERNATIVE PLOT: Uncomment the code below to plot the average temperature data ###

# Plotting the average temperature data
# plt.figure(figsize=(10, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# temp_avg.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', cbar_kwargs={'label': 'Temperature (°C)'})
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.set_extent([123.99, 126.44, 9.79, 12.93], crs=ccrs.PlateCarree())  # Set the extent to Region 8

# Adding gridlines and labels
# gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 12}
# gl.ylabel_style = {'size': 12}

# Adding title
# plt.title('Average Temperature (1961-1990)')
# plt.show()

### END OF ALTERNATIVE PLOT ###


# Computing for annual temperature average (all years)
temp_annual_avg = temp_celsius.resample(time='Y').mean()

print(temp_annual_avg)

# Calculate for the annual temperature anomalies over all years
temp_anomalies = temp_annual_avg - baseline_temp_ave

print(temp_anomalies)

# Compute the spatial average of the anomalies to reduce dimensionality
temp_anomalies_mean = temp_anomalies.mean(dim=['latitude', 'longitude'])

# Extract time and anomaly values
years = temp_anomalies['time.year'].values
anomalies = temp_anomalies_mean.values

## COMMENT THIS SECTION IF FIGURES ARE NOT NEEDED ###

# # COLORMAP FOR TEMPERATURE ANOMALIES LOOPING THROUGH ALL YEARS (see behavior of temperature anomalies throughout the years)
# # Create a directory to save the plots
# output_dir = 'temperature_anomalies_plots'
# os.makedirs(output_dir, exist_ok=True)

# # Define the levels for the contour plot
# levels = np.linspace(-0.75, 1.25, 100)

# # Loop over each year and create a plot
# for year in temp_anomalies['time.year'].values:
#     # Select the anomalies for the current year
#     anomalies_for_year = temp_anomalies.sel(time=str(year)).squeeze()
    
#     # Plotting the temperature anomalies using contourf
#     fig = plt.figure(figsize=(25., 25.), dpi=250)
#     ax = plt.axes(projection=ccrs.PlateCarree())
    
#     # Create the filled contour plot with increased number of levels for smoothness
#     norm = TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=1.25)
#     contour = ax.contourf(anomalies_for_year['longitude'], anomalies_for_year['latitude'], anomalies_for_year, 
#                           levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both', norm=norm)
    
#     # Adding coastlines and borders
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
    
#     # Adding gridlines and labels
#     gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False
#     gl.xlabel_style = {'size': 20}
#     gl.ylabel_style = {'size': 20}
    
#     # Adding colorbar
#     cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
#     cbar.set_label('Temperature Anomalies (°C)', fontsize=20)
#     cbar.ax.yaxis.set_tick_params(labelsize=20)
    
#     plt.title(f'Temperature Anomalies for {year}: (Baseline: 1961-1990)', fontsize=35)
    
#     # Save the plot to a file
#     plt.savefig(os.path.join(output_dir, f'temperature_anomalies_{year}.png'))
#     plt.close(fig)

# print(f"Plots saved to {output_dir}")


### COMMENT ABOVE SECTION IF FIGURES ARE NOT NEEDED ###



### ALTENATIVE PLOT: Uncomment the code below to plot the average temperature anomalies ###

# Plotting the average temperature anomalies
# plt.figure(figsize=(10, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# temp_anomalies_avg.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', cbar_kwargs={'label': 'Temperature Anomaly (°C)'})
# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.set_extent([123.99, 126.44, 9.79, 12.93], crs=ccrs.PlateCarree())  # Set the extent to Region 8

# Adding gridlines and labels
# gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 12}
# gl.ylabel_style = {'size': 12}

# Adding title
# plt.title('Average Temperature Anomalies (1950-2024)')
# plt.show()


### END OF ALTERNATIVE PLOT ###


# Testing for trend in temperature anomalies

import pymannkendall as mk
from matplotlib.colors import TwoSlopeNorm

# Compute the Mann-Kendall test for trend
result = mk.original_test(anomalies)

# Print the results
print(f"MK Test Results:")
print(f"  Trend: {result.trend}")
print(f"  H: {result.h}")
print(f"  P-value: {result.p:.3f}")
print(f"  Z: {result.z}")
print(f"  Tau: {result.Tau}")
print(f"  S: {result.s}")
print(f"  Var(S): {result.var_s}")
print(f"  Slope: {result.slope}")

# Compute the trend line
slope = result.slope
intercept = anomalies.mean() - slope * years.mean()
trend_line = slope * years + intercept

# Plot the temperature anomalies with the trend line
plt.figure(figsize=(20, 10), dpi=1200)
bars = plt.bar(years, anomalies, color=['blue' if anomaly < 0 else 'red' for anomaly in anomalies], linestyle='--')
plt.plot(years, trend_line, color='black', linewidth=2, label=f'Trend line (slope = {slope:.4f}) °C/year')

# Add custom legend for bars
blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Negative Anomalies')
red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Positive Anomalies')
plt.legend(handles=[blue_patch, red_patch, plt.Line2D([0], [0], color='black', lw=2, label=f'Trend line (slope = {slope:.4f}) °C/year')], loc='upper left', fontsize=15)

# Create custom legend entry for the MK test results
custom_legend = Line2D([0], [0], color='white', label=f'P-value: {result.p:.3f}\nMK Statistic: {result.z:.3f}')

# Adjust the legend position to avoid overlap
plt.legend(handles=[blue_patch, red_patch, 
                    plt.Line2D([0], [0], color='black', lw=2, label=f'Trend line (slope = {slope:.4f}) °C/year'),
                    custom_legend], loc='upper left', fontsize=15)

plt.xlabel('Year', fontsize=20)
plt.ylabel('Temperature Anomaly (°C)', fontsize=20)
plt.title('Annual Temperature Anomalies (1950-2024)', fontsize=25)
plt.grid(True)
plt.show()

# Determine the linear regression equation
print(f"Regression equation: Temperature Anomaly = {slope:.4f} * Year + {intercept:.4f}")

# Project into the future
future_year = 2050
future_temp = slope * future_year + intercept

print(f"Projected temperature anomaly in 2050: {future_temp:.4f} °C")


# Create a colormap of the temperature anomalies for the years 1950, 1975, 2000, and 2024
years_to_plot = [1950, 1975, 2000, 2024]
fig, axs = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=1200)

# Define the levels for the contour plot
levels = np.linspace(-0.75, 1.25, 100)

for ax, year in zip(axs.flat, years_to_plot):
    # Select the anomalies for the current year
    anomalies_for_year = temp_anomalies.sel(time=str(year)).squeeze()
    
    # Create the filled contour plot with increased number of levels for smoothness
    contour = ax.contourf(anomalies_for_year['longitude'], anomalies_for_year['latitude'], anomalies_for_year, 
                          levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both', vmin=-0.75, vmax=1.25)
    
    # Adding coastlines and borders
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Adding gridlines and labels
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    
    # Add a legend-like title within the subplot at the upper-right
    ax.text(0.95, 0.95, f'{year}', transform=ax.transAxes, fontsize=15, ha='right', va='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

# Adjust layout to reduce spaces between plots
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Adding colorbar with normalization to center zero
norm = TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=1.25)
contour = ax.contourf(anomalies_for_year['longitude'], anomalies_for_year['latitude'], anomalies_for_year, 
                      levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both', norm=norm)

# Adding colorbar
cbar = fig.colorbar(contour, ax=axs, orientation='vertical', pad=0.02, aspect=40, shrink=0.85)
cbar.set_label('Temperature Anomalies (°C)', fontsize=25)
cbar.ax.tick_params(labelsize=20)

# Set the colorbar ticks
cbar.set_ticks(np.arange(-0.75, 1.26, 0.25))

# Name and show the plot
plt.suptitle('Temperature Anomalies for Selected Years', fontsize=35, y=0.910)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
plt.show()
