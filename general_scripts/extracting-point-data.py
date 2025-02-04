import xarray as xr
import pandas as pd
import numpy as np

# Path to dataset file
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp.grib'

# Reading the file
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting necessary data
prcp = ds['tp']

# Latitude and longitude variables
lat_prcp = ds['latitude'].values
lon_prcp = ds['longitude'].values

# Setting the latitude and longitude of specific place in Region 8
lat_real = 11.28
lon_real = 125.06

# Calculating the nearest indices
ymin = np.abs(lat_prcp - lat_real)
yloc = ymin.argmin()
xmin = np.abs(lon_prcp - lon_real)
xloc = xmin.argmin()

# Extracting the data at the nearest indices
point_data = prcp.isel(latitude=yloc, longitude=xloc).values

# Printing the data to an excel spreadsheet
# If point_data is a single value, convert it to a list
if not isinstance(point_data, (list, np.ndarray)):
    point_data = [point_data]

# Create a DataFrame
df = pd.DataFrame(point_data, columns=['Precipitation'])

# Write the DataFrame to an Excel file
output_file = 'extracted_point_data.xlsx'
df.to_excel(output_file, index=False)

print(f"Data written to {output_file}")