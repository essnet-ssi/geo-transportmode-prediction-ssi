"""
Options for the transport mode prediction scripts.

"""

### General ###
events_fp = "data/events_vv_features.csv"  # location of the events data
location_fp = "data/geolocations_prepped_2223.csv"  # location of the gps observations data
osm_folder = "data/osm_poi_nl_d/"  # prepared osm poi data will be saved here
event_prepped_fp = "data/ssi_training_data/20250313_events_features.csv"
traintest_fp = "data/ssi_training_data/"
model_fp = 'code_for_ssi/20250313_decision_tree_ssi.pickle'  # location of the pre-trained decision tree model
vis_fp = 'output/models/20250313_decision_tree_ssi.pdf'
events_open_fp = "data/events_vv_features_UU.csv"
locations_open_fp = "data/locs_UU_anonymized_raw.csv"
events_open_prepped_fp = "data/20250327_events_features_open.csv"
min_observations = 10  # Events with less than this number of observations will be excluded.
max_duration = 10*60*60  # Unit: seconds. Events may at most be 10 hours long.

### Open Street Map features ###
osm_path = ["data/netherlands-latest.osm.pbf", 
                 "data/baden-wuerttemberg-latest.osm.pbf",
                 "data/niedersachsen-latest.osm.pbf",
                 "data/nordrhein-westfalen-latest.osm.pbf"]

buffer_m = 25  # in meters, buffer around tracks
poi_selection = {  # instructions on which osm poi categories to include for count variables. "variable_name": {"category": ["tag"]} 
    "bus_station": {"amenity": ["bus_station"]},
    "bus_stop": {"highway": ["bus_stop"]},
    "railway": {"railway": ["rail"]},  
    "light_rail": {"railway": ["light_rail"]}, 
    "subway": {"railway": ["subway"]}, 
    "tram": {"railway": ["tram"]}, 
    "busway": {"highway": ["busway"]},
    "tram_stop": {"railway": ["tram_stop"]},
    "railway_halt": {"railway": ["halt"]},
    "railway_station": {"railway": ["station"]},
    "bicycle": {"route": ['bicycle']}
}

# Note: the functions calculating proximity statistics assume that type=network and route is specified. 
poi_selection_proximity = { # instructions on which osm poi categories to include for proximity variables. "variable_name": {"category": ["tag"]}
    "bus_route": {'type': ['network'], "route": ['bus']},    
    "metro_route": {'type': ['network'], "route": ['subway']},
    "train_route": {'type': ['network'], "route": ['train']},
    "tram_route": {'type': ['network'], "route": ['tram']}
}
num_chunks = 4  # number of chunks in which to split the events data frame for parallel calculation
# Note: the number of event is assumed to be greater than num_chunks 
buffer_osm_m = 25  # in meters, buffer for selecting which osm_routes are included

### Training decision tree options ###
test_ratio = 0.3  # This part of the data will be used for testing. 1-test_ratio will be used for training.
random_seed = 123  # Random seed for reproducible results.
columns_to_exclude = ['user_id', 'label', 'label_stop', 'fase', 'VariantCode',  # These columns will be excluded from training the model.
                      'invitedstudyduration', 'label_track_grouped', 'event_id',
                      'start_time', 'end_time', 'n_geolocations', 
                      'Unnamed: 0', # artefact from concatenating data
                      'trip_duration_mins' # artefact from events_open
                      ]  
# n_geolocations is from the originally observed geolocations, whereas num_data_points is calculated after all preprocessing
columns_to_factorize = ['day_of_week', 'is_weekend']  # These columns will be transformed into categorical data.
param_grid = {  # Define the hyperparameter grid
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 25, 50],
    'min_samples_leaf': [5, 10, 15, 20, 25],  # note the count of observations in the smallest class (18 trams for our training data)
    'criterion': ['gini', 'entropy'],
    'max_features': [1, 3, 5, 7] 
}