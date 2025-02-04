import pygmt

# Define the region of interest
region = [119.00, 125.00, 8.00, 14.00]  # [West, East, South, North]

# Create a figure
fig = pygmt.Figure()

# Plot the map
fig.basemap(region=region, projection="M6i", frame=True)
fig.coast(shorelines=True, water="skyblue", land="lightgray")

# Load the topography data
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)

# Plot the topography
fig.grdimage(grid=grid, cmap="geo", shading=True)

# Add a color bar to show the elevation level
fig.colorbar(frame='af+l Elevation (m)', position='JMR+o1.5c/0c+w15c/0.5c+v')

# Show the plot with topography and color bar
fig.show()
