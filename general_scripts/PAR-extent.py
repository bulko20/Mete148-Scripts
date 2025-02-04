import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create a figure and add a map projection
fig = plt.figure(figsize=(20, 16))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set the extent of the map (longitude and latitude) 
ax.set_extent([110, 140, 0, 30], crs=ccrs.PlateCarree()) # the extent is in the format West, East, South, North

# Add features to the map
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')

# Add gridlines
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# Set the font size of the labels
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}

### Plotting PAR

# Define the coordinates of the PAR boundary
boundary_coords = [
    (115, 5), (115, 15), (120, 21), (120, 25), (135, 25), (135, 5), (115, 5)
]

# Extract the longitude and latitude from the coordinates
lon, lat = zip(*boundary_coords)

# Plot the boundary line
ax.plot(lon, lat, color='red', linewidth=2, transform=ccrs.PlateCarree())

### Plotting a specific region bounded by broken lines
### Uncomment if needed

# Define the new boundary coordinates 
# new_boundary_coords = [
#     (121, 9), (121, 12), (124, 12), (124, 9), (121, 9) # these coordinates plot a boundary box for Region 6 and Negros Island Region
# ]

# # Extract the longitude and latitude from the new coordinates
# new_lon, new_lat = zip(*new_boundary_coords)

# # Plot the new boundary line with broken lines
# ax.plot(new_lon, new_lat, color='red', linestyle='--', linewidth=2, transform=ccrs.PlateCarree())


# Display the map
plt.show()