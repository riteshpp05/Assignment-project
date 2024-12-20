README

Overview

This project demonstrates a machine learning pipeline for training and evaluating a Random Forest Classifier on the Iris dataset. The pipeline includes data preprocessing steps such as imputing missing values and encoding target labels, as well as hyperparameter tuning using GridSearchCV. The pipeline is dynamically configured based on a JSON configuration file.

Steps Included in the Code

1. Configuration Setup

The configuration for the pipeline is loaded from a JSON file (json_data.json).

Key elements extracted include:

Target Column: The column to predict (e.g., species).

Prediction Type: Type of prediction (e.g., classification or regression).

Feature Handling: Details about how features should be processed.

Feature Reduction: The method and number of features to keep.

Algorithms: List of algorithms and the selected one for training.

2. Data Loading

The Iris dataset is loaded from a CSV file (iris.csv).

The dataset is split into features (X) and the target column (y).

The target labels are encoded into numeric values using LabelEncoder.

3. Preprocessing

Numeric Columns: Missing values are imputed with the mean.

Categorical Columns: Missing values are imputed with the most frequent value.

A ColumnTransformer is used to apply these transformations separately for numeric and categorical columns.

4. Feature Engineering

Interactions and feature generation are dynamically handled based on the configuration file:

Linear Interactions

Polynomial Interactions

Explicit Pairwise Interactions

5. Feature Reduction

Tree-based feature reduction is performed if specified in the configuration file, keeping the top n features based on importance.

6. Model Selection

The selected algorithm (Random Forest Classifier in this case) is identified from the configuration file.

7. Hyperparameter Tuning

GridSearchCV is used to optimize hyperparameters for the Random Forest Classifier:

n_estimators: Number of trees in the forest.

max_depth: Maximum depth of the tree.

min_samples_split: Minimum samples required to split an internal node.

8. Pipeline Building

A Pipeline is created that combines:

The preprocessor (imputation for numeric and categorical columns).

The estimator (either the best model from GridSearchCV or the default Random Forest Classifier).

9. Model Training and Evaluation

The pipeline is trained on the dataset.

Predictions are made using the trained pipeline.

Model accuracy is evaluated using accuracy_score.

Key Outputs

Accuracy of the Model: Displays the performance of the trained Random Forest Classifier.

Requirements: Python 3.x

Libraries: pandas, scikit-learn, json

How to Run

Ensure you have the necessary libraries installed:
pip install pandas scikit-learn

Place the json_data.json file (configuration) and iris.csv file (dataset) in the specified paths.

Run the Python script.

View the output to see the target column, selected model, and accuracy.

Example Output:

Target Column: species

Prediction Type: classification

Linear Interactions: []

Polynomial Interactions: []

Explicit Pairwise Interactions: []

Performing tree-based feature reduction, keeping 5 features.

Selected Model: RandomForestClassifier

Accuracy of the model: 0.97
