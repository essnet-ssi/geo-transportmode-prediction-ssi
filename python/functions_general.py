import pandas as pd
pd.options.mode.chained_assignment = None
from pyrosm import get_data, OSM
import shapely as shapely
from copy import deepcopy


def get_event_id(events, location_df):
    """
    Adds event-based variables to every gps observation in location_df that matches an event for every event in events.
    Event based variables: label_track, label_stop, label, event_id, starttime_ts, endtime_ts. Of those, label_track and event_id
    are used in later stages for transport mode prediction.
    
    :param events: events data set
    :param location_df: gps locations data set
    :return: updated events and locations data sets
    """
    # Create new columns in location_df
    location_df['label_track'] = None
    location_df['event_id'] = None
    location_df['starttime_ts'] = None
    location_df['endtime_ts'] = None
    
    # Convert columns to datetime
    location_df['timestamp'] = pd.to_datetime(location_df['timestamp'], errors='coerce')
    events['start_time'] = pd.to_datetime(events['start_time'], errors='coerce')
    events['end_time'] = pd.to_datetime(events['end_time'], errors='coerce')

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
            label_track = row['label_track']
            event_id = row['event_id']
            
            # Create a mask for df_geo_user where timestamp falls within the interval
            mask = (df_geo_user['timestamp'] >= start) & (df_geo_user['timestamp'] <= end)
            
            # Assign the label and unique value where the mask is True
            location_df.loc[df_geo_user.index[mask], 'label_track'] = label_track
            location_df.loc[df_geo_user.index[mask], 'event_id'] = event_id
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

def select_events(events, min_observations, max_duration):
    """
    Selects events with a minimum number of observations that do not exceed a maximum duration. 
    Applying this before feature creation will reduce running time. 
    
    :param events: Dataframe with events
    :param min_observations: Minimum number of observations
    :param max_duration: Maximum duration (seconds)
    :return: Dataframe with remaining events events
    """

    len_a = len(events.index)  # original number of events
    events = events.loc[events["num_data_points"] >= min_observations]  # filter on minimum number of observations
    events = events.loc[events["trip_duration_secs"] <= max_duration]  # filter on maximum duration of event
    len_b = len(events.index)  # number of selected events

    # Print how many events were removed
    print("Removed "+str(len_a-len_b) + " events due to insufficient observations or exceeding max_duration. There are " + str(len_b) + " remaining events.")
    
    return events
