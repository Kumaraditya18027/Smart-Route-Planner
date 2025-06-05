import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load pollution data
df = pd.read_csv('../../data/processed/processed_historical_pollution_data.csv')

# Create GeoDataFrame from coordinates
geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Smaller bounding box for central Madrid (adjust if needed)
north, south, east, west = 40.43, 40.40, -3.68, -3.72
bbox = (north, south, east, west)
G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Function to calculate road density
def get_road_density(point, edges_gdf, buffer_m=300):
    buffer = point.buffer(buffer_m / 111000)  # Approximate conversion: meters to degrees
    nearby_roads = edges_gdf[edges_gdf.intersects(buffer)]
    return len(nearby_roads)

# Compute road density
gdf['road_density'] = gdf['geometry'].apply(lambda x: get_road_density(x, edges))

# Save to CSV
gdf.drop(columns=['geometry']).to_csv('../../data/final/historical_pollution_data.csv', index=False)
print("✅ Road density added and saved to historical_pollution_data.csv")
