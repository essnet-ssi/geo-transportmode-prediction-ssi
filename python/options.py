"""
Options for the transport mode prediction scripts.

"""

### General ###
events_fp = "data/events_vv_features.csv"  # location of the events data
location_fp = "data/geolocations_prepped_2223.csv"  # location of the gps observations data
model_fp = 'ODiN_Analysis/code_for_ssi/20250224_decision_tree_ssi.pickle'  # location of the pre-trained decision tree model
event_prepped_fp = "data/ssi_training_data/20250224_events_osm_no_gps.csv"
traintest_fp = "data/ssi_training_data/"
vis_fp = 'output/models/decision_tree_ssi.pdf'
min_observations = 10  # Events with less than this number of observations will be excluded.
max_duration = 10*60*60  # Unit: seconds. Events may at most be 10 hours long.

### Open Street Map features ###
osm_path = "data/netherlands-latest.osm.pbf"  # location of the osm data
# TODO: accept list of osm_path  for multiple files
buffer_m = 25  # in meters, twice this will be the actual buffer (it is applied once to tracks and once to osm)
poi_selection = {  # instructions on which osm poi categories to include. "variable_name": {"category": ["tag"]} 
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

### Training decision tree options ###
test_ratio = 0.3  # This part of the data will be used for testing. 1-test_ratio will be used for training.
random_seed = 123  # Random seed for reproducible results.
columns_to_exclude = ['user_id', 'label', 'label_stop', 'fase', 'VariantCode',  # These columns will be excluded from training the model.
                      'invitedstudyduration', 'label_track_grouped', 'event_id',
                      'start_time', 'end_time']  
columns_to_factorize = ['day_of_week', 'is_weekend']  # These columns will be transformed into categorical data.
param_grid = {  # Define the hyperparameter grid
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 25, 50],
    'min_samples_leaf': [5, 10, 15, 20, 25],  # note the count of observations in the smallest class (18 trams for our training data)
    'criterion': ['gini', 'entropy'],
    'max_features': [1, 3, 5, 7] 
}