import geopandas as gpd
import pandas as pd
import numpy as np

# Define bounding box for Madrid (adjust for your city)
min_lat, max_lat = 40.31, 40.49  # ~20km height
min_lon, max_lon = -3.84, -3.60  # ~20km width
grid_spacing = 0.0009  # ~100m (1 degree ≈ 111km, so 0.0009 ≈ 100m)

# Generate grid
latitudes = np.arange(min_lat, max_lat, grid_spacing)
longitudes = np.arange(min_lon, max_lon, grid_spacing)
grid = [(lat, lon) for lat in latitudes for lon in longitudes]

# Create DataFrame
grid_df = pd.DataFrame(grid, columns=['latitude', 'longitude'])

# Save to CSV
grid_df.to_csv('../../data/coordinates_grid.csv', index=False)
print(f"Generated {len(grid_df)} coordinates and saved to data/coordinates_grid.csv")