import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches

# Importing the dataset
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp.grib'

# Reading the file
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting total precipitation data and converting units from m to mm
total_prcp = ds['tp'] * 1000

# Getting the mean of total precipitation and grouping by month for extraction purposes
mean_total_prcp = total_prcp.mean(dim='time')
monthly_clusters_prcp = total_prcp.groupby('time.month').mean(dim='time')

# Convert the DataArray to a 2D array (flatten the spatial dimensions)
data = mean_total_prcp.values.reshape(-1, 1)

# Assigning number of clusters (k)
k = 2

# Apply k-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(data)

# Reshape the cluster labels to the original grid shape
labels_2d = labels.reshape(mean_total_prcp.shape)

# Define colors for clusters
colors = ['blue', 'red', 'green', 'purple', 'orange']
cmap = plt.cm.colors.ListedColormap(colors[:k])

# Plotting the results using contour plot with cartopy
fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add contour plot
contour = ax.contourf(ds['longitude'], ds['latitude'], labels_2d, levels=np.arange(k+1)-0.5, cmap=cmap, extend='both')

# Add geographical features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

# Set extent (optional, adjust to your region)
ax.set_extent([ds['longitude'].min(), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()], crs=ccrs.PlateCarree())

# Add title and labels
ax.set_title('K-means Clustering of Total Precipitation')
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
    cluster_values = monthly_clusters_prcp.where(cluster_mask, drop=True)
    monthly_mean = cluster_values.mean(dim=['latitude', 'longitude'])
    monthly_means.append(monthly_mean)

# Convert the list to a DataArray for easier plotting
monthly_means_da = xr.concat(monthly_means, dim='cluster')

# Defining months names
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Plot the monthly averaged values for each cluster
plt.figure(figsize=(20, 10), dpi=900)
for i in range(k):
    plt.plot(monthly_means_da['month'], monthly_means_da.sel(cluster=i), label=f'Cluster {i+1}', color=colors[i], linewidth=3)

plt.xlabel('Month', fontsize=18)
plt.ylabel('Rainfall (mm)', fontsize=16)
plt.legend(fontsize=18, loc='upper left')
plt.grid(True)
plt.xticks(ticks=range(1, 13), labels=months, rotation=45, fontsize=16)
plt.show()


### UNCOMMENT BELOW IF NEEDED

# # Create a DataFrame to export to CSV
# df = pd.DataFrame({
#     'longitude': lon_flat,
#     'latitude': lat_flat,
#     'temperature': data.flatten(),
#     'cluster': labels
# })

# # Export the DataFrame to a CSV file
# df.to_csv(r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\.py files\clustered_totalprcp_data.csv', index=False)
