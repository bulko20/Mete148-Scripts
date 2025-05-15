import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import xarray as xr
import pymannkendall as mk

# Path to the dataset file
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp.grib'

# Open the dataset
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting the dataset
temp = ds['t2m']

# Converting temp into Celsius and resampling to an annual frequency
temp_annual = (temp - 273.15).resample(time='Y').mean()

# Computing the average for all latitude and longitudes to create a time series (turning into 1D)
temp_1d = temp_annual.mean(dim=['latitude', 'longitude'])

# Extract the time (years) and temperature values
years = temp_1d['time'].dt.year.values
temperature = temp_1d.values

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(years, temperature)

# Using MK test to detect trends in the data
result = mk.original_test(temperature)

print(f"MK Test Results:")
print(f"  Trend: {result.trend}")
print(f"  H: {result.h}")
print(f"  P-value: {result.p:.3f}")
print(f"  Z: {result.z}")
print(f"  Tau: {result.Tau}")
print(f"  S: {result.s}")
print(f"  Var(S): {result.var_s}")
print(f"  Slope: {result.slope}")

# Projecting into the future
future_years = np.arange(2025, 2101)
future_temperature = slope * future_years + intercept

# Create a table of values for every 20 years
years_table = np.arange(2040, 2101, 20)
temperature_table = slope * years_table + intercept

# Plotting the observed (with trend line) and projected temperature
plt.figure(figsize=(20, 10), dpi=900)
plt.plot(years, temperature, label='Observed Temperature (1950-2024)', alpha=0.8, linewidth=2)
plt.plot(years, slope * years + intercept, label='Trend Line', color='red', linestyle='--', linewidth=3)
plt.plot(future_years, future_temperature, label='Projected Temperature', linestyle='--', color='green', linewidth=3)
plt.axvline(x=2025, color='black', label='Projection Start Year (2025)', linewidth=3)

# Annotate the plot with projected values for specific years
for year, temp in zip(years_table, temperature_table):
    plt.annotate(f'{temp:.2f}°C', xy=(year, temp), xytext=(year, temp - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=20, ha='center')

plt.xlabel('Year', fontsize=25)
plt.ylabel('Temperature (°C)', fontsize=25)

# Adding regression equation to the legend
regression_label = f'Regression equation: {slope:.4f} * Year + {intercept:.4f}'
plt.legend(loc='upper left', fontsize=20)
plt.text(0.05, 0.77, regression_label, transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')

plt.title('Temperature Trend and Projection (1950-2100)', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.show()
