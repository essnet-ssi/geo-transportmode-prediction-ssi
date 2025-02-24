"""
Train a Decision Tree model for transport mode prediction on events data with GPS-based and OSM-based features.

In the events data frame: the column `label_track` is the variable we wish to predict. 

"""

# imports
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing, model_selection, metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle



def get_train_test(events, test_ratio, columns_to_factorize, columns_to_exclude, random_seed):
    """
    ## Data preparations
    Some data preparations are required before applying the model. This is applied 
    before the train/test split to ensure the same label encoding for both train and 
    test sets. Adjust the lists of variables in the options to change which variables 
    are subject to which preprocessing step.

    ## Split train-test set
    Constraints: users should be in either train or test set, not in both. Balance 
    out the transport mode labels as evenly as possible.
    For every user, we assign a dominant mode of transportation: their most frequent 
    `label_track`. Then, the events are split into train and test set by stratifying 
    over user and dominant label. One problem in the original training data was that 
    the label tram occured only once as dominant label. This results in an error in 
    the train/test split. To avoid this error, a grouped label was introduced 
    `label_track_grouped` where bus/metro/tram were grouped, and this was used for 
    the stratification. After the train/test split, we revert back to the ungrouped 
    \version of `label_track`.

    ## Prepare train and testset variables
    Remove columns that are not the target variable and will not be known for a new 
    model. For example: the variable `label_track_grouped` from above contains 
    valuable information on the `label_track` the model should predict. Since 
    `label_track_grouped` will not be known for unseen unlabelled data, we should 
    remove it from the data shown to the model. Another example is `user_id` which 
    should be removed to avoid the model overfitting for patterns within users. The 
    columns can be chosen in the options in `columns_to_exclude`.

    :param events: Events data
    :param test_ratio: This part of the events data will be reserved for testing only
    :return: Prepared splits of train and test data 
    """

    ## Data preparations (specific for the decision tree model)
    # encode strings to factors
    le = preprocessing.LabelEncoder()  # initialize labelencoder
    for colname in columns_to_factorize:
        events[colname] = le.fit_transform(events[colname])

    # Replace infinite values with 2* max (non-inf) value and replace -infinite values with 0
    for colname in events.columns.tolist():
        if pd.api.types.is_numeric_dtype(events[colname]):
            maxvalue = 2*np.ma.masked_invalid(events[colname]).max()
            events[colname] = events[colname].replace(np.inf, maxvalue)
            events[colname] = events[colname].replace(-np.inf, 0)

    ## Split train-test set
    # Group users by their dominant label_track
    user_label_groups = events.groupby('user_id')['label_track_grouped'].agg(lambda x: x.value_counts().idxmax()).reset_index()  # most frequent label for each user

    # Rename to label_track
    user_label_groups.rename(columns={'label_track_grouped': 'dominant_label'}, inplace=True)  # new column name: dominant_label

    # Split user_ids into train and test sets with stratification based on the dominant label
    train_users, test_users = train_test_split(
        user_label_groups['user_id'],  # this variable will be used to incidate the training and testing sets
        test_size=test_ratio,  # % for testing (adjust in options)
        random_state=random_seed,  
        stratify=user_label_groups['dominant_label']  # Stratification: ensure label_track proportions
    )

    # Filter the original dataset based on train and test users split
    train_data = events[events['user_id'].isin(train_users)]  # select train set
    test_data = events[events['user_id'].isin(test_users)]  # select test set

    ## Prepare train and testset variables
    train_data = train_data.drop(columns_to_exclude, axis=1)
    test_data = test_data.drop(columns_to_exclude, axis=1)

    # Prepare the train data into `train_X` (features) and `train_y` (target variable) and remove the target 
    # variable from `test_data` as well.

    # split train_data into input and target variables
    train_X = train_data.drop('label_track', axis=1)
    train_y = train_data['label_track']

    # drop target variable from test_data
    test_X = test_data.drop('label_track', axis=1)
    test_y_true = test_data['label_track']

    return train_X, train_y, test_X, test_y_true

def train_decision_tree_gs(train_X, train_y, param_grid, random_seed):
    """
    Using a parameter grid, search for the best decision tree model for the given
    training data.
    
    :param train_X: Input features of the training set
    :param train_y: Target variable of the training set
    :param param_grid: Value ranges for parameters
    :return: Trained model

    """
    # Define the decision tree model
    my_tree = tree.DecisionTreeClassifier(random_state=random_seed)

    # Perform grid search with cross-validation
    grid_search = model_selection.GridSearchCV(my_tree, param_grid, cv=5, scoring='balanced_accuracy') 
    grid_search.fit(train_X, train_y)

    # Return the model trained for the most optimal parameter values
    trained_model = grid_search.best_estimator_
    
    return trained_model

def dt_feature_importance(trained_model):
    """
    Returns the feature importance list for the trained decision tree model.
    """
    # Get feature importances from the best model
    importances = trained_model.feature_importances_

    # Create a DataFrame to sort and display feature names with their importance
    feature_importance_df = pd.DataFrame({
        'Feature': trained_model.feature_names_in_,
        'Importance': importances
    })

    # Filter out features with importance > 0 (these were used in the tree)
    used_features_df = feature_importance_df[feature_importance_df['Importance'] > 0]

    # Sort features by importance (highest importance first)
    used_features_df = used_features_df.sort_values(by='Importance', ascending=False)

    # Print the features used in the tree
    return used_features_df

def visualize_dt(trained_model, save_loc = None):
    """
    Visualize the trained decision tree model in a plot. The plot will be saved
    if save_loc is specified.
    """

    plt.figure(figsize=(55, 20))
    tree.plot_tree(trained_model, 
            filled=True, 
            feature_names= trained_model.feature_names_in_, 
            class_names=trained_model.classes_, 
            rounded=True, 
            fontsize=8)  # Adjust the font size for better visibility

    plt.title("Decision Tree Visualization", fontsize=18)

    if save_loc:
        plt.savefig(save_loc)
    plt.show()

def evaluate_dt(trained_model, test_X, test_y_true):
    # apply model  
    test_y_pred = trained_model.predict(test_X) 

    # construct confusion matrix
    label_names = sorted(test_y_true.unique())
    cm = metrics.confusion_matrix(test_y_true, test_y_pred, labels=label_names)
    cm_data = pd.DataFrame(cm)
    cm_data.columns = label_names
    cm_data.index = label_names

    # measures of fit (based on confusion matrix)
    # accuracy multiclass
    accuracy_multi = cm.diagonal()/cm.sum(axis=1)
    
    # accuracy overall
    accuracy_all = metrics.accuracy_score(test_y_true, test_y_pred)

    # balanced accuracy
    balanced_acc = metrics.balanced_accuracy_score(test_y_true, test_y_pred)
    
    # various metrics per class
    class_report = metrics.classification_report(test_y_true, test_y_pred, target_names=label_names)

    return {"confusion_matrix": cm_data,
            "accuracy_perclass": accuracy_multi,
            "accuracy_overall": accuracy_all,
            "balanced_acc": balanced_acc,
            "class_report": class_report}

def dt_to_text(trained_model):
    """
    Print the decision nodes and leaf nodes of the trained_model decision tree. The output of this function was used in the documentation for the SSI project.

    :param trained_model: Trained decision tree model
    """

    n_nodes = trained_model.tree_.node_count
    children_left = trained_model.tree_.children_left
    children_right = trained_model.tree_.children_right
    feature = trained_model.tree_.feature
    threshold = trained_model.tree_.threshold
    values = trained_model.tree_.value
    feature_names = trained_model.feature_names_in_
    classnames = trained_model.classes_

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with values=[{valueclass}].".format(
                    space=node_depth[i] * "\t", node=i, valueclass = ", ".join([str(classnames[val_idx]) + ": " +str(np.around(val, 3)) for val_idx, val in enumerate(values[i][0]) if val>0]) ,
                )
            )
            
        else:
            print(
                "{space}node={node} is a split node with value=[{valueclass}]: "
                "go to node {left} if {feature_name} <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature_name=feature_names[feature[i]],
                    threshold=threshold[i],
                    right=children_right[i],
                    valueclass = ", ".join([str(classnames[val_idx]) + ": " +str(np.around(val, 3)) for val_idx, val in enumerate(values[i][0]) if val>0]),
                )
            )