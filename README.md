# Construction_Analysis

This project investigates factors contributing to fatalities on construction sites in New York City. Data was collected and merged from three public sources NYC OpenData (construction incidents), DOB Permit Issuance Dataset, and OpenWeatherMap Historical API (hourly weather readings) to build a custom dataset of 1,253 records with 27 features. The dataset was preprocessed through duplicate removal, median/mode imputation, IQR-based outlier treatment, SMOTE for class imbalance, standardization, and one-hot encoding. Models were trained to classify whether a given incident results in a fatality, with incremental learning applied to handle data arriving in batches without full retraining. Exploratory data analysis includes correlation heatmaps, geospatial fatality mapping, and environmental feature distributions.

====================================================================
📁 PROJECT STRUCTURE
====================================================================

NOTEBOOKS
--------------------------------------------------------------------
CS-23024.ipynb
    Main notebook. Contains all five algorithm implementations
    (Logistic Regression, KNN, SVM, Naive Bayes, ANN), incremental
    learning pipelines, evaluation metrics, graphs, and test cases.
    Run this to reproduce all results.

Data_Preprocessing.ipynb
    Data cleaning, feature engineering, standardisation, and SMOTE
    balancing. Run this first if starting from raw data.


DATA FILES
--------------------------------------------------------------------

Raw / Preprocessed
-------------------
construction_weather_data_final.csv
    Original merged dataset combining construction incident records
    with hourly weather data. Starting point for all preprocessing.

Preprocessed_Construction_Inciden....csv
    Output of the preprocessing pipeline. Cleaned and feature-
    engineered data before standardisation or balancing.

Train / Test Splits
--------------------
Standardized_Train.csv
    Standardised (z-score) training set — 80% split.
    Input to the SMOTE balancing step.

Standardized_Test.csv
    Standardised test set — held-out 20% split. Never used during
    training. Used for all final model evaluations.

Standardized_Train_70.csv
    Alternative 70% training split for sensitivity experiments.

Standardized_Test_30.csv
    Corresponding 30% test split, paired with Standardized_Train_70.

Balanced Training Sets
-----------------------
Balanced_Training_Set_Only.csv
    SMOTE-balanced version of the 80% training split.
    Primary training file used by all five algorithms.

Balanced_Training_Set_Only_70.csv
    SMOTE-balanced version of the 70% training split.
    Used for alternative split experiments.

Final Test Sets (real class distribution)
------------------------------------------
Test_Set_Only.csv
    Held-out test set matched to Balanced_Training_Set_Only.
    Preserves real class distribution. Primary file for all
    classification reports and metric tables.

Test_Set_Only_30.csv
    Held-out test set for the 70/30 split experiments.


INCREMENTAL LEARNING BATCHES
--------------------------------------------------------------------
batch_32_rows_1.csv to batch_32_rows_5.csv
    Five 32-row batches used to demonstrate and test the incremental
    learning capability of the final algorithm implementation.


SAVED MODEL
--------------------------------------------------------------------
svm_model2.pkl
    Serialised (pickle) file of the best-performing SVM model (M2).
    Load this to run predictions without retraining.
