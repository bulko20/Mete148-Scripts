import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches

# Importing the dataset
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp.grib'

# Reading the file
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting total precipitation data
temp = ds['t2m']

# Converting units from K to °C
temp_cels = temp - 273.15

# Extracting longitude and latitude coordinates
lon = ds['longitude'].values
lat = ds['latitude'].values

# Ensure lon and lat are 2D arrays matching the spatial dimensions of temp
lon_2d, lat_2d = np.meshgrid(lon, lat)

# Getting the mean of temperature and grouping by month for extraction purposes
mean_temp_cels = temp_cels.mean(dim='time')
monthly_clusters_temp = temp_cels.groupby('time.month').mean(dim='time')

# Convert the DataArray to a 2D array (flatten the spatial dimensions)
data = mean_temp_cels.values.reshape(-1, 1)

# Assigning number of clusters (k)
k = 2

# Apply k-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(data)

# Reshape the cluster labels to the original grid shape
labels_2d = labels.reshape(mean_temp_cels.shape)

# Define colors for clusters
colors = ['blue', 'red', 'green', 'purple', 'orange']
cmap = plt.cm.colors.ListedColormap(colors[:k])

# Plotting the results using contour plot with cartopy
fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add contour plot
contour = ax.contourf(lon_2d, lat_2d, labels_2d, levels=np.arange(k+1)-0.5, cmap=cmap, extend='both')

# Add geographical features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

# Set extent (optional, adjust to your region)
ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

# Add title and labels
ax.set_title('K-means Clustering of Temperature')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Create custom legend
legend_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i+1}') for i in range(k)]
ax.legend(handles=legend_patches, loc='upper left')

# Show the plot
plt.show()

# Extract the monthly averaged values for each cluster
monthly_means = []
for i in range(k):
    cluster_mask = xr.DataArray((labels_2d == i), dims=["latitude", "longitude"])
    cluster_values = monthly_clusters_temp.where(cluster_mask, drop=True)
    monthly_mean = cluster_values.mean(dim=['latitude', 'longitude'])
    monthly_means.append(monthly_mean)

# Convert the list to a DataArray for easier plotting
monthly_means_da = xr.concat(monthly_means, dim='cluster')

# Defining months names
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Plot the monthly averaged values for each cluster
plt.figure(figsize=(20, 10))
for i in range(k):
    plt.plot(monthly_means_da['month'], monthly_means_da.sel(cluster=i), label=f'Cluster {i+1}', color=colors[i])

plt.title('Monthly Averaged Temperature for Each Cluster')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend(loc='upper left', fontsize=14)
plt.grid(True)
plt.xticks(ticks=range(1, 13), labels=months, rotation=45, fontsize=14)
plt.show()

### UNCOMMENT AS NEEDED
### USED AS AN INPUT FILE FOR QGIS

# # Create a DataFrame to export to CSV
# df = pd.DataFrame({
#     'longitude': lon_2d.flatten(),
#     'latitude': lat_2d.flatten(),
#     'temperature': data.flatten(), # flatten data to 1D array
#     'cluster': labels
# })

# # Export the DataFrame to a CSV file
# df.to_csv(r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\.py files\clustered_temperature_data.csv', index=False)
