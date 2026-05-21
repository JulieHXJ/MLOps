# Titanic Survival Prediction with Logistic Regression

This project implements a Logistic Regression model to predict passenger survival in the Titanic dataset.

The goal of the project is not only to train a model with `scikit-learn`, but also to understand the internal logic of Logistic Regression by implementing it manually with NumPy.

## Workflow

The project follows a basic machine learning workflow:

1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Select features and target
4. Split the data into training and test sets
5. Train a Logistic Regression model with `scikit-learn`
6. Evaluate the model with classification metrics
7. Implement Logistic Regression from scratch with NumPy
8. Compare the custom implementation with the `scikit-learn` model

## Model Evaluation

The model is evaluated using:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- ROC AUC

These metrics provide a more complete understanding of model performance than accuracy alone.

## Manual Implementation

The NumPy implementation includes:

- feature standardization
- sigmoid function
- binary cross-entropy loss
- gradient descent
- prediction and accuracy functions
- loss history tracking

The Logistic Regression process can be summarized as:

```text
features -> linear score -> sigmoid probability -> class prediction