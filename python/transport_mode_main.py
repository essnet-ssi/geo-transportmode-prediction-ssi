"""
Main script for transport mode prediction.

For transparency, the code for training the model is provided in train_decision_tree.py, 
however, since the data on which the original model was trained is not publicly available,
the code will not run automatically. Instead, the pickle file for the trained model is 
provided so it can be applied.

"""
from options import *  # file with all options
from osm_features import *
from gps_features import *
from train_decision_tree import *
import pickle



# Prepare events data
if False:
    # load datasets
    chunk = pd.read_csv(location_fp, chunksize=10000)
    location_df = pd.concat(chunk)
    chunk = pd.read_csv(events_fp, chunksize=10000)
    events = pd.concat(chunk)

    # add osm_features
    events = add_osm_features(events, location_df, osm_path, poi_selection, buffer_m)
    # add gps_features
    events = add_gps_features(events, location_df,min_observations, max_duration)

    events.to_csv(event_prepped_fp, index=False)
elif False:
    # Save time on the prepocsesing if this has already been saved
    chunk = pd.read_csv(event_prepped_fp, chunksize=10000)
    events = pd.concat(chunk)

    
# Load decision tree model
if False:
    # Train model from scratch
    train_X, train_y, test_X, test_y_true = get_train_test(events, test_ratio, columns_to_factorize, columns_to_exclude, random_seed)

    trained_model = train_decision_tree_gs(train_X, train_y, param_grid, random_seed)
    test_y_pred = trained_model.predict(test_X)

    # save the train and test dataframes for future use:
    train_X.to_csv(traintest_fp+"train_X.csv")
    #train_y.to_csv(traintest_fp+"train_y.csv")
    test_X.to_csv(traintest_fp+"test_X.csv")
    #test_y_true.to_csv(traintest_fp+"test_y_true.csv")
    #test_y_pred.to_csv(traintest_fp+"test_y_pred.csv")

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

else:
    # Otherwise, load a pretrained model
    with open(model_fp, "rb") as model_file:
        trained_model = pickle.load(model_file)

# inspect model
print(dt_feature_importance(trained_model))  
visualize_dt(trained_model, save_loc=vis_fp)
dt_to_text(trained_model)

