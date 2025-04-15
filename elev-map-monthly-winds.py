import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

### Function for reading RGB color values from a file
def read_rgb_file(rgb_file_path):
    with open(rgb_file_path, 'r') as open_file:
        rgb_lines = open_file.readlines()

    converted_colors = []
    for rgb_line in rgb_lines:
        # Split the line into components
        rgb_components = rgb_line.split()

        # Convert the RGB values to the range 0-1 and add to the list
        r, g, b = [int(c) / 255 for c in rgb_components[1:4]]
        converted_colors.append((r, g, b))

    return converted_colors

### End of function


# Reading the RGB color values for the colormap
rgb_file_path = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\.py files\colormaps\windbeaufort_kmh.rgb'

# Read the RGB file and create a custom colormap
rgb_colors = read_rgb_file(rgb_file_path)
custom_cmap = mcolors.ListedColormap(rgb_colors, name='custom_wind_cmap')


# Importing the dataset
grib_file = r'C:\Users\HP\Dropbox\PC\Desktop\College\4th Year\1st Sem\Research 1\Data\Actual\all-years-wind-data.grib'
ds = xr.open_dataset(grib_file, engine='cfgrib')

# Extracting all necessary data needed
uwnd = ds['u10']
vwnd = ds['v10']

# Taking the mean of the u- and v-wind components on a monthly basis
mean_uwnd = uwnd.groupby('time.month').mean(dim='time')
mean_vwnd = vwnd.groupby('time.month').mean(dim='time')

# Computing for the actual averaged wind vector on a monthly basis
mean_wind_speed = mpcalc.wind_speed(mean_uwnd.values * units('m/s'), mean_vwnd.values * units('m/s'))

# List of months
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# Plotting the results using contour plot with cartopy
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10.5, 20.5), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True, dpi=900)

# Creating a custom colormap and normalizing the colorbar
bounds = [0, 2, 4, 6, 8, 10]
norm = Normalize(vmin=0, vmax=10)

for i, ax in enumerate(axes.flat):
    month_idx = (i + 11) % 12  # Start from December
    cs = ax.contourf(mean_uwnd.longitude, mean_uwnd.latitude, mean_wind_speed[month_idx], cmap=custom_cmap, vmin=0, vmax=10, transform=ccrs.PlateCarree())
    ax.coastlines()

    # Adding streamlines for wind direction
    x, y = np.meshgrid(mean_uwnd.longitude, mean_uwnd.latitude)
    uwind = mean_uwnd[month_idx].values
    vwind = mean_vwnd[month_idx].values
    ax.streamplot(x, y, uwind, vwind, color='k', density=0.5, arrowsize=2.5, arrowstyle='-|>', transform=ccrs.PlateCarree())

    ax.set_extent([mean_uwnd.longitude.min().item(), mean_uwnd.longitude.max().item(), mean_uwnd.latitude.min().item(), mean_uwnd.latitude.max().item()], crs=ccrs.PlateCarree())

# Adding a colorbar
cax = fig.add_axes([1.05, 0.15, 0.02, 0.7])
colorbar = ColorbarBase(cax, cmap=custom_cmap,
                        norm=norm, extend='max',
                        ticks=np.arange(0, 11, 1))
colorbar.set_label('Wind speed (m/s)', fontsize=30)
colorbar.ax.tick_params(labelsize=25)

plt.show()
