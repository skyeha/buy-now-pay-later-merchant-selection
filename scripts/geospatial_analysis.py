import folium
import geopandas as gpd
from pyspark.sql import functions as F, SparkSession
from IPython.display import display


def create_consumer_map(geojson,consumer_group_by_postcode, identifier, value, key):
    m = folium.Map(location=[-25.2744, 133.7751], zoom_start=4)
    # Add Choropleth layer to the map
    folium.Choropleth(
        geo_data=geojson,
        name='choropleth',
        data=consumer_group_by_postcode,
        columns=[identifier, value],
        key_on='feature.properties.' + key,
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name= value
    ).add_to(m)
    display(m)