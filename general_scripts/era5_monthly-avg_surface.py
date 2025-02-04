import os
import logging
import cdsapi

# Creating the client
c = cdsapi.Client()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configuring the output directory
output_dir = r'insert\file\path'

# Creating a loop to download the data
for year in range(1951, 1952): 

    # Log information for year being downloaded
    logging.info(f'Downloading data for {year}...')

    # Defining the output file
    output_file = os.path.join(output_dir, f'era5_{year}.grib')

    # Data retrieval
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'mean_sea_level_pressure', 'total_precipitation',
            ],

            'year': str(year),
            'month': [
                '01', '02', '03', 
                '04', '05', '06', 
                '07', '08', '09', 
                '10', '11', '12',
            ],

            'time':[
                '00:00',
            ],

            'area': [
                12.95, 124.15, 9.85, 126.20,  # North, West, South, East
            ],

            'format': 'grib',
        },
        output_file
    )

logging.info('Download complete.')
