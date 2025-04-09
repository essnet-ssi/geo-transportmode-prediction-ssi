"""
Main script for transport mode prediction.

For transparency, the code for training the model is provided in train_decision_tree.py, 
however, since the data on which the original model was trained is not publicly available,
the code will not run automatically. Instead, the pickle file for the trained model is 
provided so it can be applied.

"""
from options import *  # file with all options
from functions_general import *
from osm_features import *
from gps_features import *
from train_decision_tree import *
import pickle



# Prepare events data for training
if False:
    # load datasets
    chunk = pd.read_csv(location_fp, chunksize=10000)
    location_df = pd.concat(chunk)
    chunk = pd.read_csv(events_fp, chunksize=10000)
    events = pd.concat(chunk)

    # Add event_id as link between events and location_df
    events = events.rename(columns={'cls_start_ts': 'start_time', 'cls_end_ts': 'end_time'})  #Rename columns for merge
    events["event_id"] = range(len(events))  # Generate unique event id's vor every row in events 
    
    # Assign unique event_id to each event (to identify which gps observations belong to which event)
    location_df = get_event_id(events, location_df)
    
    # Only keep gps observations with a known event
    location_df = location_df[location_df['event_id'].notnull()]
    
    # add osm_features
    events = add_osm_features(events, location_df, osm_path, poi_selection, 
                              poi_selection_proximity, buffer_m, buffer_osm_m, 
                              osm_folder, num_chunks)
    # add gps_features
    events = add_gps_features(events, location_df, min_observations, max_duration)

    # save the events data with newly added features
    events.to_csv(event_prepped_fp, index=False)
elif False:
    # Save time on the prepocsesing if this has already been saved
    chunk = pd.read_csv(event_prepped_fp, chunksize=10000)
    events = pd.concat(chunk)


# Train or load decision tree model
if False:
    # Train model from scratch (only possible with plenty of input data)
    train_X, train_y, test_X, test_y_true = get_train_test(events, test_ratio, columns_to_factorize, columns_to_exclude, random_seed)

    trained_model = train_decision_tree_gs(train_X, train_y, param_grid, random_seed)
    test_y_pred = trained_model.predict(test_X)

    # save the train and test dataframes for future use:
    train_X.to_csv(traintest_fp+"train_X.csv")
    test_X.to_csv(traintest_fp+"test_X.csv")

    print(train_X.columns.tolist())

    print("n_leaves: "+str(trained_model.get_n_leaves()))
    print("depth: "+str(trained_model.get_depth()))
    print("params: "+str(trained_model.get_params()))

    with open(model_fp, 'wb') as handle:
        pickle.dump(trained_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Model evaluation on train set (included in the SSI documentation) 
    evaluation_metrics = evaluate_dt(trained_model, train_X, train_y)
    print("Evaluation on train set:")
    print("   Confusion matrix: ")
    print(evaluation_metrics["confusion_matrix"])
    print("   Accuray (per class): " + str(evaluation_metrics["accuracy_perclass"]))
    print("   Accuray (over all): " + str(evaluation_metrics["accuracy_overall"]))
    print("   Balanced accuracy: " + str(evaluation_metrics["balanced_acc"]))
    print("   Class report: ")
    print(evaluation_metrics["class_report"])

    # Model evaluation on test set (included in the SSI documentation) 
    evaluation_metrics = evaluate_dt(trained_model, test_X, test_y_true)
    print("Evaluation on test set:")
    print("   Confusion matrix: ")
    print(evaluation_metrics["confusion_matrix"])
    print("   Accuray (per class): " + str(evaluation_metrics["accuracy_perclass"]))
    print("   Accuray (over all): " + str(evaluation_metrics["accuracy_overall"]))
    print("   Balanced accuracy: " + str(evaluation_metrics["balanced_acc"]))
    print("   Class report: ")
    print(evaluation_metrics["class_report"])    

elif True:
    # Load a pretrained model
    with open(model_fp, "rb") as model_file:
        trained_model = pickle.load(model_file)

if True:  
    # Inspect trained model
    print(dt_feature_importance(trained_model))  
    visualize_dt(trained_model, save_loc=vis_fp)
    dt_to_text(trained_model)

# Prepare Open Geodata from SSI
if True:
    # Apply trained model to open SSI geodata (events_open and locations_open)
    # For applying the model to your own data, use the structure of the below code to ensure 
    # 1) variable names match those expected by the model
    # 2) each variable containsn the values corresponding to the expected unit

    # Read data:
    # Save time on the prepocsesing if this has already been saved
    chunk = pd.read_csv(events_open_fp, chunksize=10000)
    events_open = pd.concat(chunk)

    chunk = pd.read_csv(locations_open_fp, chunksize=10000)
    locations_open = pd.concat(chunk)

    # Transform some variables if necessary: 
    events_open["trip_duration_secs"] = events_open["trip_duration_mins"]*60 

    # Rename columns
    events_open = events_open.rename(columns={'start_timestamp': 'start_time', # old_name: new_name
                                              'end_timestamp': 'end_time',
                                              'mode': 'label_track'})

    # Add event_id as link between events and location_df
    events_open["event_id"] = range(len(events_open))  # Generate unique event id's for every row in events 

    # Assign unique event_id to each event (to identify which gps observations belong to which event)
    locations_open = get_event_id(events_open, locations_open)
    
    # Only keep gps observations with a known event (increases running time)
    locations_open = locations_open[locations_open['event_id'].notnull()]

    # add osm_features
    events_open = add_osm_features(events_open, locations_open, osm_path, poi_selection, 
                              poi_selection_proximity, buffer_m, buffer_osm_m, 
                              osm_folder, num_chunks)
    
    # add gps_features
    events_open = add_gps_features(events_open, locations_open, min_observations, max_duration)

    events_open_all['label_track'] = events_open_all['label_track'].replace({'Bike': 'fiets',
                                        'Walk': 'voet',
                                        'Ferry': "ferry",
                                        'Train': 'trein',
                                        'Metro': 'metro',
                                        'Tram': 'tram',
                                        'Bus': 'bus'})
    
    # save the prepared open events data with newly added features
    events_open.to_csv(events_open_prepped_fp, index=False)

# Load Open Geodata from SSI
elif True:
    # Save time on the prepocsesing if this has already been saved
    chunk = pd.read_csv(events_open_prepped_fp, chunksize=10000)
    events_open = pd.concat(chunk) 

    print("n events: "+str(len(events_open.index)))

    # prepare variables of events so they match the expected model input:
    events_open = feature_preparation_dt(events_open, columns_to_factorize)

# Apply model to Open Geodata from SSI
if True:
    # Convert to test data format 
    # We want to use all open data as test data, so put test_ratio=0.5 (arbitrary value) and simply concatenate afterwards
    # Not performing this step is not an option, because we need to prepare the variables in exactly the same way

    # Prepare train and testset variables
    columns_to_exclude_selection = [c for c in columns_to_exclude if c in events_open.columns]
    test_data_open = events_open.drop(columns_to_exclude_selection, axis=1)
    
    # drop target variable from test_data
    test_X_open = test_data_open.drop('label_track', axis=1)
    test_y_true_open = test_data_open['label_track']

    # Ensure features are in the expected order by the model
    features_input = trained_model.feature_names_in_  # get feature names and order of expected input
    test_X_open = test_X_open[features_input]  # reorder test_X to match expected input
   
    # Model evaluation on test set (included in the SSI documentation) 
    evaluation_metrics = evaluate_dt(trained_model, test_X_open, test_y_true_open)
    print("Evaluation on open test set:")
    print("   Confusion matrix: ")
    print(evaluation_metrics["confusion_matrix"])
    print("   Accuray (per class): " + str(evaluation_metrics["accuracy_perclass"]))
    print("   Accuray (over all): " + str(evaluation_metrics["accuracy_overall"]))
    print("   Balanced accuracy: " + str(evaluation_metrics["balanced_acc"]))
    print("   Class report: ")
    print(evaluation_metrics["class_report"])   
    
    
    
