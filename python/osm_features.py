"""
This python script contains functions for OpenStreetMap-based feature creation for transportmode prediction.
"""

from functions_general import *
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from pyrosm import get_data, OSM
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import shapely as shapely
from copy import deepcopy
from pathlib import Path
import copy 
from functools import partial
from multiprocessing import Pool


def create_geometry(group):
    """
    Function to create geometry for each track
    
    :param group: GPS observations for which we want to create a geometry
    :return: Geometry for the group of GPS observations

    """

    points = list(zip(group['lon'], group['lat']))  # put in right format
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
   
    # Add feature: duration of event
    location_df = add_cls_length_seconds(location_df)

    # Remove duplicate gps observations
    location_df = location_df.drop_duplicates(subset=['user_id','event_id' ,'lon', 'lat', 'timestamp'], keep='first')
    # GeoDataFrame
    geometries = location_df.groupby('event_id').apply(create_geometry)  # Group by cls_uuid and create geometries
    events_gdf = gpd.GeoDataFrame(geometries, columns=['geometry'], crs="EPSG:4326").reset_index()  # Convert to GeoDataFrame
    events_gdf = events_gdf.to_crs(epsg=3395)  # Create buffer around each track

    # This line is giving problems:
    events_gdf['buffered_geometry'] = events_gdf['geometry'].apply(lambda x: x.buffer(buffer_m))  # Buffer the geometries by buffer_m meters 
    
    # Add feature: Calculate the area of the buffered_geometry. This allows us to calculate normalized counts later. 
    events_gdf['total_buffer_area'] = events_gdf['buffered_geometry'].area
    # Select wanted variables:
    events_buffered = events_gdf[["event_id", "buffered_geometry", "total_buffer_area"]]
    events_linestring = events_gdf[["event_id", "geometry"]]
    
    return events, events_buffered, events_linestring

def load_poi_data(variable_name, osm_path, poi_selection, osm_folder):
    """
    Load osm poi data for the desired variable. Or make selection if this is the first time. 
    All data will be saved in the folder /data/osm_poi/
    Note that the variable_name is used to determine if this is a new or previously saved variable.
    Changing the tags in poi_selection without changing the variable_name, will not be noticed by 
    this function and result in loading the variable_name with the old tag settings. In that case,
    make sure to manually delete the saved file for variable_name in /data/osm_poi/ so the osm_poi
    data will be freshly loaded.
    
    :param variable_name: The variable name used to reference to features based on this variable name.
    :param osm_path: A list of filepaths that contain osm data. All of these files will be loaded.
    :param poi_selection: Dictionary defined in options.py that contains specific tags for each variable name.
    :param osm_folder: Folder in which the prepared poi selections are stored. Make sure to change folder
        name when you change the selection of osm filepaths. 
    :return: The selected poi corresponding to variable_name.
    """
    
    filename = osm_folder + "osm_poi_" + str(variable_name) + ".csv"
    
    if not Path(filename).exists():
        # This is the first time osm_poi data for this variable_name is requested, so it has not yet been saved.
        # Load the osm Point of Interest data for the tag previously specified to belong to this variable name
        print("Loading osm poi data for "+ str(variable_name) + ". This may take a while. Saving afterwards.")
        
        osm_poi_list = []  # we'll store the separate osm_poi in this list

        for osm_path_tmp in osm_path:

            # Initialize the OSM parser object
            osm_data = OSM(osm_path_tmp)

            osm_poi_tmp = osm_data.get_pois(custom_filter=poi_selection[variable_name])
            try:
                print("   Found " + str(len(osm_poi_tmp)) + " points of interest in "+str(osm_path_tmp)+".")
                osm_poi_list.append(osm_poi_tmp)
            except TypeError:
                # There were no points of interest of the specified type in the OSM file for this region
                print("   Found 0 points of interest in "+str(osm_path_tmp)+".")
                
        # add all partial osm_poi_tmp in the list into a single file:
        osm_poi = pd.concat(osm_poi_list, axis=0)
        osm_poi.to_csv(osm_folder + "osm_poi_"+variable_name+".csv", index=False)

    if Path(filename).exists():
        # This will always be run, even if osm_poi was just freshly created. This is because the "geometry" 
        # column will behave differently when read from csv or when freshly created. 
        print("Reading previously saved osm poi data for "+ str(variable_name) + ".")
        chunk = pd.read_csv(osm_folder + "osm_poi_"+variable_name+".csv", chunksize=10000)
        osm_poi = pd.concat(chunk)
    
    return osm_poi

def add_poi_count_features(events, variable_name, events_buffered, osm_path, poi_selection, osm_folder):
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
    osm_poi_tmp = load_poi_data(variable_name, osm_path, poi_selection, osm_folder)  # get the poi data for this filter(variable_name)

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

def calculate_distance_from_track(matches, event_row):
    """
    This function calculates the distances of every point in one track, to the nearest route in a set of matched OSM routes.
    
    :param matches: Set of selected OSM routes
    :param event_row: Row from event data set, must contain geometry
    :return: List of distances
    """
    geometry_track = event_row['geometry']  # non-buffered (should come from events_linestring)
    #geometry_poi = shapely.line_merge([match["geometry_poi"] for match in matches])  # merge all matched osm routes into a MultiLineString
    geometry_poi = shapely.line_merge(matches["geometry_poi"])  # merge all matched osm routes into a MultiLineString
    
    # Extract points from the LineString (geometry_track)
    points_track = [Point(coord) for coord in geometry_track.coords]  
    
    # Calculate the distance from each point to the nearest point on the poi geometry
    distances = [point.distance(geometry_poi) for point in points_track]
    
    # Calculate and return multiple statistics on the distances
    return distances

def get_distance_features(event_row, matched_gdf, variable_name_prefix):
    """
    Returns various features about the distance between one track (event) and a set of selected OSM routes.
    First, the distances are calculated with calculate_distance_from_track(). Second, they are summarised
    into various features.
    
    :param event_row: Row from event data set, must contain geometry
    :param matched_gdf: Dataframe with selected OSM routes for all events
    :param variable_name_prefix: String to include in the final feature names, to later distinguish between features.
    :return: Distance features
    """
    # Prepare the arguments to calculate distances
    event_id_tmp = int(event_row["event_id"])
    matched_selection = matched_gdf.loc[matched_gdf["event_id"] == event_id_tmp]  # select relevant matched OSM routes

    if len(matched_selection.index) > 0:
        # At least one matching route exists
        # Calculate distances
        distances = calculate_distance_from_track(matched_selection, event_row)
        # Calculate distance statistics
        distance_stats = {'event_id': event_row['event_id'], 
                            variable_name_prefix + '_min_distance': np.min(distances), 
                            variable_name_prefix + '_max_distance': np.max(distances),
                            variable_name_prefix + '_mean_distance': np.mean(distances), 
                            variable_name_prefix + '_std_distance': np.std(distances)}
    
    else: 
        # There is no match, so return NaN values instead
        distance_stats = {'event_id': event_row['event_id'],
            variable_name_prefix + '_min_distance': np.NaN, 
            variable_name_prefix + '_max_distance': np.NaN, 
            variable_name_prefix + '_mean_distance': np.NaN, 
            variable_name_prefix + '_std_distance': np.NaN}

    return distance_stats

def apply_distance_features(chunk, matched_gdf, variable_name_prefix):
    """
    Helper function to apply get_distance_features() in parallel.
    
    :param chunk: Part of events data 
    :param matched_gdf: Dataframe with selected OSM routes for all events
    :param variable_name_prefix: String to include in the final feature names, to later distinguish between features.
    :return: Dataframe of results from get_distance_features()
    """
    
    # Apply the function and convert the dictionary output into a DataFrame
    #results = chunk.apply(calculate_distance_to_track(variable_name_prefix=variable_name_prefix), axis=1)
    results = chunk.apply(lambda x: get_distance_features(x, matched_gdf=matched_gdf, 
                                                          variable_name_prefix=variable_name_prefix), axis=1)
    
    #return pd.DataFrame([r[0] for r in results], index=[0])
    return pd.DataFrame(results.tolist(), index=chunk.index)

def parallel_distance_calculation(df, matched_gdf, num_chunks=4, variable_name_prefix=""):
    """
    Helper function to apply get_distance_features() in parallel.
    
    :param df: Events data 
    :param matched_gdf: Dataframe with selected OSM routes for all events
    :param num_chunks: Number of processes that should be run in parallel
    :param variable_name_prefix: String to include in the final feature names, to later distinguish between features.
    :return: Dataframe of results from get_distance_features()
    """
    
    # Function to apply calculation in parallel
    # Split the DataFrame into chunks
    chunk_size = len(df.index) // num_chunks

    if chunk_size < len(df.index):
        # avoid error in range() 
        chunk_size=1

    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df.index), chunk_size)]

    print("chunk size: "+str(chunk_size))
    
    # Initialize Pool and parallelize the calculation
    with Pool(num_chunks) as pool:
        result = pool.map(partial(apply_distance_features, matched_gdf=matched_gdf, variable_name_prefix=variable_name_prefix), chunks)
    
    # Combine the results
    distance_stats = pd.concat(result, axis=0)
    
    return distance_stats

def add_poi_prox_features(events, variable_name, events_buffered, events_linestring, osm_path, poi_selection_proximity, osm_folder, num_chunks,
                          buffer_osm_m):
    """
    Adds for every event in events, proximity features based on distance to preselected osm points of interest. 

    :param events: Dataframe with events
    :param variable_name: What name to use for the new proximity variables
    :param events_buffered: events with buffered geometries around the track (parameter: buffer_m)
    :param osm_data: preselected osm point of interest data
    :return: Dataframe with events, with four new variables: min_distance, max_distance, mean_distance, std_distance
    """

    events_prox = copy.deepcopy(events)  # in this copy of events, we'll add the osm counts

    # Load poi data for the desired variable
    osm_poi_tmp = load_poi_data(variable_name, osm_path, poi_selection_proximity, osm_folder)  # get the poi data for this filter(variable_name)

    if len(osm_poi_tmp.index) == 0:
        print("No osm poi were found for the variable "+ variable_name + " so no features were added.")
        return events

    # Ensure the right format of the osm geodata
    osm_poi_tmp = osm_poi_tmp[osm_poi_tmp['type']=='route']  # requirement for distance calculations
    print("Number of osm poi for variable "+variable_name + ": " + str(len(osm_poi_tmp.index)))
    osm_poi_tmp['geometry_poi'] = osm_poi_tmp['geometry'].apply(shapely.wkt.loads) # Ensure the geometry is in shapely format
    osm_poi_gdf = gpd.GeoDataFrame(osm_poi_tmp, geometry='geometry_poi', crs="EPSG:4326")  # Convert to GeoDataFrameusing WGS-84 crs (lat-lon)
    osm_poi_gdf = osm_poi_gdf.to_crs(epsg=3395)
    # Add buffer to poi geometry for matching only. The original geometry_poi will be used for distance calculation.
    osm_poi_gdf['buffered_geometry_poi'] = osm_poi_gdf['geometry_poi'].apply(lambda x: x.buffer(buffer_osm_m)) 
    
    # Duplicate the geometry column in events, so one can be kept in the upcoming merge with an informative name
    events_buffered['geometry_track_buffered'] = events_buffered['buffered_geometry']  

    print("Searching for matches between poi and track.")
    print("   len(osm_poi_gdf.index): "+str(len(osm_poi_gdf.index)))
    print("   len(events_buffered.index): "+str(len(events_buffered.index)))
    
    # Check for intersection between osm and events
    matched_gdf = gpd.sjoin(osm_poi_gdf.set_geometry('buffered_geometry_poi'), # change this to buffered_geometry if you are buffering both sides
                    events_buffered.set_geometry('buffered_geometry'), 
                    how='inner', 
                    predicate='intersects')
    
    print("# matches in matched_gdf for variable "+variable_name + ": " + str(len(matched_gdf.index)))
    
    if len(matched_gdf.index) == 0:
        print("No osm poi were matched with routes for "+ variable_name + " so no features were added.")
        return events

    # calculate distance features 
    distance_stats = parallel_distance_calculation(df=events_linestring, matched_gdf=matched_gdf, 
                                          num_chunks = num_chunks, variable_name_prefix=variable_name)
    print(distance_stats.head())
    print(distance_stats.columns.tolist())
    events_prox = events_prox.merge(distance_stats, on="event_id", how="left") # add to events_prox

    return events_prox
                          
def add_osm_features(events, location_df, osm_path, poi_selection, poi_selection_proximity, 
                     buffer_m, buffer_osm_m, osm_folder, num_chunks):
    """
    This function adds OSM-based features to an event data set. 
    
    :param events: Dataframe with events
    :param location_df: Dataframe with GPS observations per event
    :param osm_path: Filepath to OSM data
    :param poi_selection: OSM tags to be counted and variable names in which to store the counts. Dictionary of the structure {"variable_name": {"category": ["tag"]},...}
    :param poi_selection_proximity: OSM tags to be used for proximity features. Dictionary of the structure {"variable_name": {"category": ["tag"]},...}
    :param buffer_m: Size of buffer in meters
    :param buffer_osm_m: Size of OSM buffer in meters
    :param osm_folder: Filepath to folder where intermediate OSM poi files can be stored
    :param num_chunks: Number of processes that should be run in parallel
    :return: Events dataframe with OSM-based features
    """
    print("Preparing gps geometries")
    # Add buffered GPS routes to events
    events, events_buffered, events_linestring = prepare_gps_data(events=events, location_df=location_df, buffer_m=buffer_m) 
 
    # Create OSM-based counting features
    # the osm tags to be counted and variable names in which to store the counts are listed in poi_selection
    # merge the selection into one dict to give to function as argument
    for varname in poi_selection.keys():
        print("Creating counting features for "+varname)
        events = add_poi_count_features(events, varname, events_buffered, osm_path, poi_selection, osm_folder)
    
    # Create OSM-based proximity features
    for varname in poi_selection_proximity.keys():
        print("Creating proximity features for "+varname)
        events = add_poi_prox_features(events=events, variable_name=varname, events_buffered=events_buffered, 
                                    events_linestring=events_linestring,
                                    osm_path=osm_path, poi_selection_proximity=poi_selection_proximity, 
                                    osm_folder=osm_folder, num_chunks=num_chunks,
                                    buffer_osm_m=buffer_osm_m)
        
        events.to_csv("data/ssi_training_data/20250314_events_"+varname+".csv", index=False)
        
    return events

    
