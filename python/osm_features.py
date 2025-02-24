"""
Docstring for osm_features.py
"""

import pandas as pd
pd.options.mode.chained_assignment = None
from pyrosm import get_data, OSM
import geopandas as gpd
from shapely.geometry import Point, LineString
import shapely as shapely
from itertools import compress
from copy import deepcopy
from pathlib import Path
import copy 

def convert_columns_to_datetime(df, columns_to_convert):
    """
    Convert time columns to datetime.
    
    :param df: Dataframe of which the columns should be converted.
    :param columns_to_convert: Specification of which columns to convert.
    :return: Dataframe with converted columns.
    """

    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce') 
    
    return df

def get_event_id(events, location_df):
    """
    Add unique event_id for every event in events and its relevant gps observations in location_df.
    
    :param events: events data set
    :param location_df: gps locations data set
    :return: updated events and locations data sets
    """
    # Create new columns in location_df
    location_df['label_track'] = None
    location_df['label_stop'] = None
    location_df['label'] = None
    location_df['event_id'] = None
    location_df['starttime_ts'] = None
    location_df['endtime_ts'] = None

    # Iterate over each user in df_event
    # Sort on both user_id and start_time to ensure consistent event_id values 
    for user in events['user_id'].unique():
        # Subset the data for the current user
        df_event_user = events[events['user_id'] == user]
        df_geo_user = location_df[location_df['user_id'] == user]
        
        if df_event_user.empty or df_geo_user.empty:
            continue

        # Iterate over each interval in df_event_user for the current user
        for idx, row in df_event_user.iterrows():
            start = row['start_time']
            end = row['end_time']
            label = row['label']
            label_track = row['label_track']
            label_stop = row['label_stop']
            
            # Create a mask for df_geo_user where timestamp falls within the interval
            mask = (df_geo_user['timestamp'] >= start) & (df_geo_user['timestamp'] <= end)
            
            # Assign the label and unique value where the mask is True
            location_df.loc[df_geo_user.index[mask], 'label_track'] = label_track
            location_df.loc[df_geo_user.index[mask], 'label_stop'] = label_stop
            location_df.loc[df_geo_user.index[mask], 'label'] = label
            location_df.loc[df_geo_user.index[mask], 'event_id'] = row['event_id']
            location_df.loc[df_geo_user.index[mask], 'starttime_ts'] = start
            location_df.loc[df_geo_user.index[mask], 'endtime_ts'] = end

        return location_df
    
def add_cls_length_seconds(df):
    """
    Add the lenght of an event to the events dataframe.
    
    :param df: events dataframe
    :return: updated events dataframe

    """
    df['starttime_ts'] = pd.to_datetime(df['starttime_ts'], errors='coerce')
    df['endtime_ts'] = pd.to_datetime(df['endtime_ts'], errors='coerce')
    
    # Calculate the length in seconds
    df['cls_length_seconds'] = (df['endtime_ts'] - df['starttime_ts']).dt.total_seconds()

    df['cls_length_minutes'] = (df['endtime_ts'] - df['starttime_ts']).dt.total_seconds() /60

    df['cls_length_hours'] = (df['endtime_ts'] - df['starttime_ts']).dt.total_seconds()/(60*60)
    
    return df

def create_geometry(group):
    """
    Function to create geometry for each track
    
    :param group: GPS observations for which we want to create a geometry
    :return: Geometry for the group of GPS observations

    """
    points = list(zip(group['lon'], group['lat']))
    if len(points) > 1:
        # There are enough points to create a geometry
        return LineString(points)
    else:
        # This group contained only one GPS observation, so return only the single point
        return Point(points[0])
    
def prepare_gps_data(events, location_df, buffer_m):
    """
    Add geometry to the events data for the relevant gps observations
    
    :param events: dataframe with events and event_id
    :param location_df: dataframe with gps locations and event_id
    :param buffer_m: amount of meaters to be used as radius for the buffers in the geometries
    :return: events and events_buffered, the latter will contain buffered geometries
    """
    
    #Rename columns for merge
    events = events.rename(columns={'cls_start_ts': 'start_time', 'cls_end_ts': 'end_time'})
    
    # Generate unique event id's vor every row in events 
    events["event_id"] = range(len(events))

    # Convert columns to datetime
    location_df = convert_columns_to_datetime(location_df, columns_to_convert = ['timestamp', 'start_time', 'end_time'])
    events = convert_columns_to_datetime(events, columns_to_convert = ['timestamp', 'start_time', 'end_time'])

    # Assign unique event_id to each event (and identify which gps observations belong to which event)
    location_df = get_event_id(events, location_df)

    # Only keep gps observations for which we know the label
    location_df = location_df[location_df['label_track'].notnull()]

    # Add feature: duration of event
    location_df = add_cls_length_seconds(location_df)

    # Remove duplicate gps observations
    location_df = location_df.drop_duplicates(subset=['user_id','event_id' ,'lon', 'lat', 'timestamp'], keep='first')

    # GeoDataFrame
    geometries = location_df.groupby('event_id').apply(create_geometry)  # Group by cls_uuid and create geometries
    events_gdf = gpd.GeoDataFrame(geometries, columns=['geometry'], crs="EPSG:4326").reset_index()  # Convert to GeoDataFrame
    events_gdf = events_gdf.to_crs(epsg=3395)  # Create buffer around each track
    events_gdf['buffered_geometry'] = events_gdf['geometry'].apply(lambda x: x.buffer(buffer_m))  # Buffer the geometries by buffer_m meters 

    # Add feature: Calculate the area of the buffered_geometry. This allows us to calculate normalized counts later. 
    events_gdf['total_buffer_area'] = events_gdf['buffered_geometry'].area

    # Select wanted variables:
    events_buffered = events_gdf[["event_id", "buffered_geometry", "total_buffer_area"]]

    return events, events_buffered

def load_poi_data (variable_name, osm_data, poi_selection):
    """
    Load osm poi data for the desired variable. Or make selection if this is the first time. All data will be saved in the folder /data/osm_poi/
    """
    filename = "data/osm_poi/osm_poi_" + str(variable_name) + ".csv"

    if Path(filename).exists():
        print("Reading previously saved osm poi data for "+ str(variable_name) + ".")
        chunk = pd.read_csv("data/osm_poi/osm_poi_"+variable_name+".csv", chunksize=10000)
        osm_poi = pd.concat(chunk)
    
    else:
        # Load the osm Point of Interest data for the tag previously specified to belong to this variable name
        print("Loading osm poi data for "+ str(variable_name) + ". This may take a while. Saving afterwards.")
        
        osm_poi = osm_data.get_pois(custom_filter=poi_selection[variable_name])
        
        print("Found " + str(len(osm_poi)) + " points of interest.")
        osm_poi.to_csv("data/osm_poi/osm_poi_"+variable_name+".csv", index=False)
        
    return osm_poi

def add_poi_count_features(events, variable_name, events_buffered, osm_data, poi_selection):
    """
    Adds for every event in events, a count and normalized count with the number of preselected osm points of interest. 

    :param events: Dataframe with events
    :param variable_name: What name to use for the new count variables
    :param events_buffered: events with buffered geometries around the track (parameter: buffer_m)
    :param osm_data: preselected osm point of interest data
    :return: Dataframe with events, with two new variables: count and normalized count 
    """

    events_counts = copy.deepcopy(events)  # in this copy of events, we'll add the osm counts

    # Load poi data for the desired variable
    osm_poi_tmp = load_poi_data(variable_name, osm_data, poi_selection)  # get the poi data for this filter(variable_name)

    # Ensure the right format of the osm geodata
    osm_poi_tmp['geometry'] = osm_poi_tmp['geometry'].apply(shapely.wkt.loads)  # Ensure the geometry is in shapely format
    # osm_poi_tmp = osm_poi_tmp.explode(ignore_index=True, column="geometry") # if needed: to expand multilinestrings into multiple linestrings
    osm_poi_gdf = gpd.GeoDataFrame(osm_poi_tmp, geometry='geometry', crs="EPSG:4326")  # Convert to GeoDataFrameusing WGS-84 crs (lat-lon)
    osm_gdf = osm_poi_gdf.to_crs(epsg=3395)
    
    # Check for intersection between osm and events
    matched_gdf = gpd.sjoin(osm_gdf.set_geometry('geometry'), # change this to buffered_geometry if you are buffering both sides
                    events_buffered.set_geometry('buffered_geometry'), #events_buffered.set_geometry('geometry'), 
                    how='inner', 
                    predicate='intersects')
    
    # For every event, count the number of matches with the osm_poi data
    events_match_counts = matched_gdf.groupby('event_id', as_index=False).agg(
        osm_poi_count_tmp=('geometry', 'size')  # Count the number of matches, generic name (change later)
        ).rename(columns={'osm_poi_count_tmp': variable_name+"_count"}) # rename the variable name so we know which count to use

    events_counts = events_counts.merge(events_match_counts, on="event_id", how="left")  # add to events_counts

    # normalized count
    events_counts = events_counts.merge(events_buffered[["event_id", "total_buffer_area"]], on="event_id", how="left")  # get the total_buffer_area from events_buffered
    events_counts[variable_name+"_normcount"] = events_counts[variable_name+"_count"]/events_counts["total_buffer_area"]
    events_counts = events_counts.drop("total_buffer_area", axis=1)

    return events_counts

def add_osm_features(events, location_df, osm_path, poi_selection, buffer_m):
    """
    Docstring for prepare_osm_data
    
    :param osm_path: Description
    :param poi_selection: OSM tags to be counted and variable names in which to store the counts. Dictionary of the structure {"variable_name": {"category": ["tag"]},...}
    """
    # Add buffered GPS routes to events
    events, events_buffered = prepare_gps_data(events, location_df, buffer_m)

    # Initialize the OSM parser object
    osm_data = OSM(osm_path)

    # the osm tags to be counted and variable names in which to store the counts are listed in poi_selection
    # merge the selection into one dict to give to function as argument
    for varname in poi_selection.keys():
        print("Creating counting features for "+varname)
        events = add_poi_count_features(events, varname, events_buffered, osm_data, poi_selection)

    return events

    
