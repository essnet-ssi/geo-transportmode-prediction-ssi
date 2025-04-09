# Geo-transportmode-prediction-ssi

This microservice presents a transport mode prediction algorithm for smart surveys. Built using a decision tree model, it uses smartphone GPS data and OpenStreetMap (OSM) infrastructure. Initially developed with data from Statistics Netherlands, the algorithm is evaluated using public geo-data from the SSI GitHub repository. The goal is to automate mode detection, reduce manual labeling, and ease survey participation.

## Documentation

The development is described and documented in geo-transportmode-prediction-ssi_documentation.pdf.

## Overview on central functions

All required functions of this microservice can be found in the folder python.

### transport_mode_main.py

The main script for transport mode prediction that will load and run the other scripts.

### options.py

Contains all options regarding file paths, data preprocessing and model training required in the other scripts.

### functions_general.py

Contains general functions required for the transport mode prediction process.
        
### gps_features.py
    
Contains functions for gps-based feature creation for events and locations data. These features are added to the events dataframe.

### osm_features.py
   
Contains functions for osm-based feature creation for events and locations data. These features are added to the events dataframe. 

### train_decision_tree.py

Runs a grid search over a hyperparameter set to train the best decision tree model for the given data. The current best result is decision\_tree\_ssi.pickle.

 ### decision\_tree\_ssi.pickle
        
The best decision tree model for the available development data resulting from the combination of options, feature creation and model training.
