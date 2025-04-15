# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import spearmanr

# Path to the dataset files
prcp_path = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp-COMPLETE.grib'
temp_path = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp-COMPLETE.grib'

# Opening the dataset files
ds_prcp = xr.open_dataset(prcp_path, engine='cfgrib')
ds_temp = xr.open_dataset(temp_path, engine='cfgrib')

# Extracting the dataset and converting to SI units
tp_mm = ds_prcp['tp'] * 1000
temp_celsius = ds_temp['t2m'] - 273.15

# Aligning the time dimension of precipitation to match that of temperature
start_date_tp = np.datetime64('1950-01-31T00:00:00')
tp_mm = tp_mm.sel(time=slice(start_date_tp, None))
# start_date_temp = np.datetime64('1950-02-01T00:00:00')
# temp_celsius = temp_celsius.sel(time=slice(start_date_temp, None))

# Trimming the longer dataset to match the length of the shorter one
min_length = min(len(tp_mm['time']), len(temp_celsius['time']))
tp_mm = tp_mm.isel(time=slice(0, min_length))
temp_celsius = temp_celsius.isel(time=slice(0, min_length))

# Resampling the data to a monthly frequency and averaging over all latitude and longitudes
tp_mm_monthly = tp_mm.resample(time='M').mean().mean(dim=['latitude', 'longitude'])
temp_celsius_monthly = temp_celsius.resample(time='M').mean().mean(dim=['latitude', 'longitude'])

# Grouping the data by month of all years
months = np.arange(1, 13)
spearman_results = {}

for month in months:
    tp_mm_month = tp_mm_monthly.sel(time=tp_mm_monthly['time.month'] == month)
    temp_celsius_month = temp_celsius_monthly.sel(time=temp_celsius_monthly['time.month'] == month)
    
    # Extracting the values for precipitation and temperature
    precipitation = tp_mm_month.values
    temperature = temp_celsius_month.values
    
    # Ensure both arrays have the same length
    min_len = min(len(precipitation), len(temperature))
    precipitation = precipitation[:min_len]
    temperature = temperature[:min_len]
    
    # Ensure both arrays have the same length
    min_len = min(len(precipitation), len(temperature))
    precipitation = precipitation[:min_len]
    temperature = temperature[:min_len]
    
    # Ensure both arrays have the same length
    min_len = min(len(precipitation), len(temperature))
    precipitation = precipitation[:min_len]
    temperature = temperature[:min_len]
    
    # Calculating the Spearman correlation coefficient between precipitation and temperature
    r, p_value = spearmanr(precipitation, temperature)
    spearman_results[month] = (r, p_value)
    
    # Printing the correlation coefficient and p-value for each month
    print(f"Month: {month}")
    print(f"Spearman Correlation Coefficient: {r:.4f}")
    print(f"P-value: {p_value:.4f}")
    print()

# Creating scatter plots to visualize the correlation for each month
fig, axs = plt.subplots(3, 4, figsize=(24, 18), dpi=1200)
axs = axs.flatten()

for i, month in enumerate(months):
    tp_mm_month = tp_mm_monthly.sel(time=tp_mm_monthly['time.month'] == month)
    temp_celsius_month = temp_celsius_monthly.sel(time=temp_celsius_monthly['time.month'] == month)
    
    precipitation = tp_mm_month.values
    temperature = temp_celsius_month.values 
    
    r, p_value = spearman_results[month]
    
    axs[i].scatter(precipitation, temperature, color='b', alpha=0.8)
    axs[i].plot(np.unique(precipitation), np.poly1d(np.polyfit(precipitation, temperature, 1))(np.unique(precipitation)), color='r', linestyle='--', linewidth=3)
    axs[i].grid()

plt.tight_layout()
plt.show()