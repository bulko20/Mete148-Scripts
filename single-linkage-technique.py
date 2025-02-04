import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans # type: ignore
from sklearn.metrics import silhouette_score #type: ignore
import pandas as pd
import cartopy.crs as ccrs #type: ignore
import cartopy.feature as cfeature #type: ignore
import matplotlib.patches as mpatches

### FOR PRECIPITATION

# Importing the dataset
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-totalprcp.grib'

# Reading the file
ds_rain = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting total precipitation data and converting to mm
total_prcp = ds_rain['tp'] * 1000

# Computing the mean of total precipitation
mean_total_prcp = total_prcp.mean(dim='time')

# Extracting latitude and longitude value
lat = ds_rain['latitude'].values
lon = ds_rain['longitude'].values

# Ensure lon and lat are 2D arrays matching the spatial dimensions of total_prcp
lon_2d, lat_2d = np.meshgrid(lon, lat)

# Convert the DataArray to a 2D array (flatten the spatial dimensions)
data_prcp = mean_total_prcp.values.reshape(-1, 1)

# Perform single linkage clustering to create a dendrogram
Z_prcp = linkage(data_prcp, method='single')

# Calculate the cophenetic correlation coefficient
c, coph_dists = cophenet(Z_prcp, pdist(data_prcp))

# Print the cophenetic correlation coefficient
print(f'Cophenetic Correlation Coefficient for Precipitation: {c}')

# Plot full dendrogram
plt.figure(figsize=(20, 10))
dendrogram(Z_prcp, no_labels=True)
plt.title('Dendrogram for Total Precipitation', fontsize=20)
plt.ylabel('Dissimilarity', fontsize=15)
plt.show()

# Evaluate silhouette coefficient for different values of k
silhouette_scores = []
for k in range(2, 6):
    # Apply k-means clustering with the determined number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_prcp)
    labels = kmeans.labels_

    # Calculate the silhouette coefficient
    silhouette_prcp_avg = silhouette_score(data_prcp, labels)
    silhouette_scores.append((k, silhouette_prcp_avg))

# Create a DataFrame to display the silhouette coefficients
df_silhouette_prcp = pd.DataFrame(silhouette_scores, columns=['Number of Clusters (k)', 'Silhouette Coefficient (Precipitation)'])
print(df_silhouette_prcp)


### FOR TEMPERATURE

# Importing the dataset
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-temp.grib'

# Reading the file
ds_t2m = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting temperature data and converting to °C
temp_cels = ds_t2m['t2m'] - 273.15

# Computing the mean of temp
mean_temp = temp_cels.mean(dim='time')

# Extracting latitude and longitude values
lat = ds_t2m['latitude'].values
lon = ds_t2m['longitude'].values

# Ensure lon and lat are 2D arrays matching the spatial dimensions of temp
lon_2d, lat_2d = np.meshgrid(lon, lat)

# Convert the DataArray to a 2D array (flatten the spatial dimensions)
data_temp = mean_temp.values.reshape(-1, 1)

# Perform single linkage clustering to create a dendrogram
Z_temp = linkage(data_temp, method='single')

# Calculate the cophenetic correlation coefficient
c, coph_dists = cophenet(Z_temp, pdist(data_temp))

# Print the cophenetic correlation coefficient
print(f'Cophenetic Correlation Coefficient for Temperature: {c}')

# Plot full dendrogram
plt.figure(figsize=(20, 10))
dendrogram(Z_temp, no_labels=True)
plt.title('Dendrogram for Temperature', fontsize=20)
plt.ylabel('Dissimilarity', fontsize=15)
plt.show()

# Evaluate silhouette coefficient for different values of k
silhouette_scores = []
for k in range(2, 6):
    # Apply k-means clustering with the determined number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_temp)
    labels = kmeans.labels_

    # Calculate the silhouette coefficient
    silhouette_temp_avg = silhouette_score(data_temp, labels)
    silhouette_scores.append((k, silhouette_temp_avg))

# Create a DataFrame to display the silhouette coefficients
df_silhouette_temp = pd.DataFrame(silhouette_scores, columns=['Number of Clusters (k)', 'Silhouette Coefficient (Temperature)'])
print(df_silhouette_temp)


### MULTI-VARIABLE CLUSTERING ###

# Ensure lon and lat are 2D arrays matching the spatial dimensions of total_prcp
lon_2d, lat_2d = np.meshgrid(lon, lat)

# Convert the DataArray to a 2D array (flatten the spatial dimensions)
data_prcp_combined = mean_total_prcp.values.reshape(-1, 1)
data_temp_combined = mean_temp.values.reshape(-1, 1)

# Combine the data into a single array
data_combined = np.hstack((data_prcp_combined, data_temp_combined))

# Perform single linkage clustering to create a dendrogram
Z = linkage(data_combined, method='single')

# Calculate the cophenetic correlation coefficient
c, coph_dists = cophenet(Z, pdist(data_combined))

# Print the cophenetic correlation coefficient
print(f'Cophenetic Correlation Coefficient for Combined: {c}')

# Plot full dendrogram
plt.figure(figsize=(20, 10))
dendrogram(Z, no_labels=True)
plt.title('Dendrogram for Total Precipitation and Temperature', fontsize=20)
plt.ylabel('Dissimilarity', fontsize=15)
plt.show()

# Evaluate silhouette coefficient for different values of k
silhouette_scores = []
for k in range(2, 6):
    # Apply k-means clustering with the determined number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_combined)
    labels = kmeans.labels_

    # Calculate the silhouette coefficient
    silhouette_avg = silhouette_score(data_combined, labels)
    silhouette_scores.append((k, silhouette_avg))

# Create a DataFrame to display the silhouette coefficients
df_silhouette_combined = pd.DataFrame(silhouette_scores, columns=['Number of Clusters (k)', 'Silhouette Coefficient (Combined)'])
print(df_silhouette_combined)

# Choose a higher number of clusters based on domain knowledge, and silhouette coefficients of different k
k = 2

# Apply k-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data_combined)
labels = kmeans.labels_

# Flatten the 2D longitude and latitude arrays to match the data array
lon_flat = lon_2d.flatten()
lat_flat = lat_2d.flatten()

# Reshape the cluster labels to the original grid shape
labels_2d = labels.reshape(mean_total_prcp.shape)

# Define colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange']
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
ax.set_title('K-means Clustering of Total Precipitation and Temperature')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Create custom legend
legend_patches = [mpatches.Patch(color=colors[i], label=f'Cluster {i+1}') for i in range(k)]
ax.legend(handles=legend_patches, loc='upper left')

# Show the plot
plt.show()