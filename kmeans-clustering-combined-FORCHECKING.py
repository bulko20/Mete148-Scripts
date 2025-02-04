import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches

# Importing the dataset
grib_tp = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp.grib'
grib_t2m = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp.grib'

# Reading the file
ds_tp = xr.open_dataset(grib_tp, engine='cfgrib')
ds_t2m = xr.open_dataset(grib_t2m, engine='cfgrib')

# Extracting total precipitation and temperature data
prcp = ds_tp['tp'] * 1000  # convert to mm
temp = ds_t2m['t2m'] - 273.15  # convert to Celsius

# Resampling the data to monthly means
mean_tp = prcp.groupby('time.month').mean(dim='time')
mean_temp = temp.groupby('time.month').mean(dim='time')

# Flatten the data arrays and combine them
combined_data = np.hstack([mean_tp.values.reshape(-1, 12), mean_temp.values.reshape(-1, 12)])

# Apply k-means clustering
k = 2
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(combined_data)

# Reshape the cluster labels to the original grid shape
labels_2d = labels.reshape(mean_tp.shape[1:])

# Define colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange']
cmap = plt.cm.colors.ListedColormap(colors[:k])

# Plotting the results using contour plot with cartopy
fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add contour plot
contour = ax.contourf(ds_tp['longitude'], ds_tp['latitude'], labels_2d, levels=np.arange(k + 1) - 0.5, cmap=cmap, extend='both')

# Add geographical features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)

# Set extent (optional, adjust to your region)
ax.set_extent([ds_tp['longitude'].min(), ds_tp['longitude'].max(), ds_tp['latitude'].min(), ds_tp['latitude'].max()], crs=ccrs.PlateCarree())

# Add title and labels
ax.set_title('K-means Clustering of Combined Total Precipitation and Temperature')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Create custom legend
legend_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i + 1}') for i in range(k)]
ax.legend(handles=legend_patches, loc='upper left')

# Show the plot
plt.show()

# Define month names
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Extract the monthly averaged values for each cluster
monthly_means_tp = []
monthly_means_temp = []
for i in range(k):
    cluster_mask = (labels_2d == i)
    cluster_values_tp = mean_tp.where(cluster_mask, drop=True)
    cluster_values_temp = mean_temp.where(cluster_mask, drop=True)
    monthly_means_tp.append(cluster_values_tp.mean(dim=['latitude', 'longitude']))
    monthly_means_temp.append(cluster_values_temp.mean(dim=['latitude', 'longitude']))

# Convert the lists to DataArrays for easier plotting
monthly_means_tp_da = xr.concat(monthly_means_tp, dim='cluster')
monthly_means_temp_da = xr.concat(monthly_means_temp, dim='cluster')

# Plot the monthly averaged values for each cluster
plt.figure(figsize=(20, 10))
for i in range(k):
    plt.plot(monthly_means_tp_da['month'], monthly_means_tp_da.sel(cluster=i), label=f'Cluster {i + 1} Precipitation', color=colors[i], linestyle='-')
    plt.plot(monthly_means_temp_da['month'], monthly_means_temp_da.sel(cluster=i), label=f'Cluster {i + 1} Temperature', color=colors[i], linestyle='--')

plt.title('Monthly Averaged Precipitation and Temperature for Each Cluster')
plt.xlabel('Month')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# Set x-axis to month names and skew them 45°
plt.xticks(ticks=range(1, 13), labels=month_names, rotation=45)

plt.show()

# # Create a DataFrame to export to CSV
# df = pd.DataFrame({
#     'longitude': lon_flat,
#     'latitude': lat_flat,
#     'temperature': combined_data,
#     'cluster': labels
# })

# # Export the DataFrame to a CSV file
# df.to_csv(r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\.py files\clustered_combined_data.csv', index=False)