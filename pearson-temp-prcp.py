# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import pearsonr

# Path to the dataset files
prcp_path = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp.grib'
temp_path = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp.grib'

# Opening the dataset files
ds_prcp = xr.open_dataset(prcp_path, engine='cfgrib')
ds_temp = xr.open_dataset(temp_path, engine='cfgrib')

# Extracting the dataset and converting to SI units
tp_mm = ds_prcp['tp'] * 1000
temp_celsius = ds_temp['t2m'] - 273.15

# Aligning the time dimension of precipitation to match that of temperature
start_date = np.datetime64('1950-01-31T00:00:00')
tp_mm = tp_mm.sel(time=slice(start_date, None))

# Resampling the data to an annual frequency and averaging over all latitude and longitudes
tp_mm_annual = tp_mm.resample(time='Y').mean(dim=['latitude', 'longitude'])
temp_celsius_annual = temp_celsius.resample(time='Y').mean(dim=['latitude', 'longitude'])

# Extracting the time (years) and values for precipitation and temperature
years = tp_mm_annual['time'].values
precipitation = tp_mm_annual.values
temperature = temp_celsius_annual.values

# Convert years to a numerical format
years_numeric = np.array([year.astype('datetime64[Y]').astype(int) + 1970 for year in years])

# Creating plots for precipitation and temperature
fig, axs = plt.subplots(2, 1, figsize=(20, 20))

# Plot for precipitation
axs[0].plot(years_numeric, precipitation, label='Observed Precipitation', alpha=0.8)
axs[0].set_xlabel('Year', fontsize=15)
axs[0].set_ylabel('Precipitation (mm)', fontsize=15)
axs[0].set_title('Annually Averaged Total Precipitation in Region 8 (1950-2024)', fontsize=18)
axs[0].legend(loc='upper left', fontsize=15)
axs[0].grid()

# Plot for temperature
axs[1].plot(years_numeric, temperature, label='Observed Temperature', alpha=0.8)
axs[1].set_xlabel('Year', fontsize=15)
axs[1].set_ylabel('Temperature (°C)', fontsize=15)
axs[1].set_title('Annually Averaged Temperature in Region 8 (1950-2024)', fontsize=18)
axs[1].legend(loc='upper left', fontsize=15)
axs[1].grid()

plt.show()

# Calculating the Pearson correlation coefficient between precipitation and temperature
r, p_value = pearsonr(precipitation, temperature)

# Printing the correlation coefficient and p-value
print(f"Pearson Correlation Coefficient: {r:.4f}")
print(f"P-value: {p_value:.4f}")

# Creating a scatter plot to visualize the correlation between precipitation and temperature
plt.figure(figsize=(20, 10))
plt.scatter(precipitation, temperature, color='b', alpha=0.8)
plt.plot(np.unique(precipitation), np.poly1d(np.polyfit(precipitation, temperature, 1))(np.unique(precipitation)), color='r', linestyle='--', linewidth=3, label='Line of Best Fit')
plt.xlabel('Precipitation (mm)', fontsize=15)
plt.ylabel('Temperature (°C)', fontsize=15)
plt.title('Correlation between Precipitation and Temperature in Region 8 (1950-2024)', fontsize=18)
plt.text(0.05, 0.92, f'Pearson Correlation Coefficient: {r:.4f}\nP-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
plt.legend(loc='upper left', fontsize=15)
plt.grid()
plt.show()
