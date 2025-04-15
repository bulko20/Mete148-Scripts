import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import xarray as xr
import pymannkendall as mk

# Path to the dataset file
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp.grib'

# Open the dataset
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting the dataset and converting total precipitation from meters to mm
total_prcp = ds['tp'] * 1000

# Start date of the dataset
start_date = np.datetime64('1950-01-31T00:00:00')
total_prcp = total_prcp.sel(time=slice(start_date, None))

# Resampling the data to an annual frequency and computing the average for all latitude and longitudes
tp_mm_avg_ts = total_prcp.resample(time='Y').mean().mean(dim=['latitude', 'longitude'])

# Extract the time (years) and precipitation values
years = tp_mm_avg_ts['time'].dt.year.values
precipitation = tp_mm_avg_ts.values

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(years, precipitation)

# Using MK test to test for trend significance
result = mk.original_test(precipitation)

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
future_precipitation = slope * future_years + intercept

# Create a table of values for every 20 years
years_table = np.arange(2040, 2101, 20)
prcp_table = slope * years_table + intercept

# Plotting the observed (with trend line) and projected precipitation
plt.figure(figsize=(20, 10), dpi=900)
plt.plot(years, precipitation, label='Observed Precipitation (1950-2024)', alpha=0.8, linewidth=2)
plt.plot(years, slope * years + intercept, label='Trend Line', color='red', linestyle='--', linewidth=3)
plt.plot(future_years, future_precipitation, label='Projected Precipitation', linestyle='--', color='green', linewidth=3)
plt.axvline(x=2025, color='black', label='Projection Start Year (2025)', linewidth=3)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Precipitation (mm)', fontsize=15)

# Annotate the plot with projected values for specific years
for year, prcp_1 in zip(years_table, prcp_table):
    plt.annotate(f'{prcp_1:.2f} mm', xy=(year, prcp_1), xytext=(year, prcp_1 - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=14, ha='center')

# Adding regression equation to the legend
regression_label = f'Regression equation: {slope:.4f} * Year + {intercept:.4f}'
plt.legend(loc='upper left', fontsize=15)
plt.text(0.05, 0.77, regression_label, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
plt.grid()

plt.show()
