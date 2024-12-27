import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import compute_sample_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold


def model_predict(model, X_train, y_train, X_test, balance = None, weights = False):
    """
    Train the provided machine learning model and make predictions.

    Parameters:
        model: The machine learning model to be trained.
        X_train: Training feature dataset (features).
        y_train: Training target dataset (labels).
        X_test: Testing feature dataset (features for prediction).
        balance (optional): A class balancing technique.
        weights (optional): Sample weights for training the model.

    Returns:
        y_pred: Predicted class labels for the test dataset.
        y_pred_proba: Predicted probabilities for each class in the test dataset.
                      For binary classification, probabilities for the positive class are returned.
    """

    # Apply class balancing, if specified
    if balance:
        X_train, y_train = balance.fit_resample(X_train, y_train)

    # Check if sample weights are needed
    if weights is True:
        sample_weights = compute_sample_weight('balanced', y_train)
        fit = model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        fit = model.fit(X_train, y_train)

    y_pred = fit.predict(X_test)
    
    # Predict class probabilities for the test set
    y_pred_proba = fit.predict_proba(X_test)

    # Handle binary classification by returning probabilities for the positive class
    if len(np.unique(y_train)) == 2:
        y_pred_proba = y_pred_proba[:, 1]

    # Return predicted labels and probabilities
    return y_pred, y_pred_proba


def model_performance(y_test, y_pred, y_pred_proba, perf = False, conf = False, classes = None):
    """
    Given a set of predictions, it evaluates with different metrics the performance of a model in the classification of each class.

    Parameters:
        y_test: True labels of the test set.
        y_pred: Predicted class labels from the model.
        y_pred_proba: Predicted probabilities made with the trained model for the computation of the ROC-AUC Score.
        perf (optional): If True, prints performance metrics for each class.
        conf (optional): If True, displays the confusion matrix.
        classes (optional): Custom label for class 0 and 1 in binary classification when displaying the confusion matrix.

    Returns:
        precision_per_class: Precision score for each class.
        recall_per_class: Recall score for each class.
        f1_per_class: F1 score for each class.
        accuracy: Overall accuracy of the model.
        roc_auc: ROC-AUC score for binary or multiclass classification.
        conf_matr (if conf=True): Displays a confusion matrix visualization.
    """

    # Compute precision, recall, and F1 scores for each class individually
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Compute the overall accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    # Compute the ROC-AUC score
    if len(np.unique(y_test)) == 2:  # Binary classification
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    # Identify classes
    classes = sorted(set(y_test))

    # Print detailed performance metrics if `perf` is True
    if perf:
        print("Performance evaluation for each class:")

        # Print precision, recall, and F1 score for each class
        for i, cls in enumerate(classes):
            print(f"Class {cls}:")
            print(f"  Precision: {precision_per_class[i]:.5f}")
            print(f"  Recall: {recall_per_class[i]:.5f}")
            print(f"  F1 Score: {f1_per_class[i]:.5f}\n")

        # Print overall accuracy and ROC-AUC score
        print(f"Accuracy: {accuracy:.5f}")
        print(f"ROC-AUC Score: {roc_auc:.5f}")

    # Display the confusion matrix if `conf` is True
    if conf:
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        # Binary classification
        if len(np.unique(y_test)) == 2:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        # Multiclass classification
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Galaxy", "Star", "QSO"])

        # Plot the confusion matrix with a color map
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=15, pad=20)
        plt.xlabel('Predicted Class', fontsize=11)
        plt.ylabel('True Class', fontsize=11)
        plt.show()

    return precision_per_class, recall_per_class, f1_per_class, accuracy, roc_auc


def model_scores(model_name: str, model, balance: str, X_train, y_train, X_test, y_test, balancing = None, weights = False):
    """
    Train, predict, and evaluate a machine learning model, returning the performance metrics in a structured format.

    Parameters:
        model_name: The name of the machine learning model being evaluated.
        model: The machine learning model to be trained and evaluated.
        balance: A description of the class balancing method applied (or "None" if no balancing is applied).
        X_train: Training feature dataset (features).
        y_train: Training target dataset (labels).
        X_test: Testing feature dataset (features for prediction).
        y_test: True labels for the test set.
        balancing (optional): A class balancing technique.
        weights (optional): Sample weights for training the model.

    Returns:
        A single-row DataFrame containing the performance metrics for the given model.
        Includes precision, recall, F1 score for each class, accuracy, and ROC-AUC.
    """

    # Generate predictions and probability scores using the provided model
    y_pred, y_pred_proba = model_predict(model, X_train, y_train, X_test, balancing, weights)

    # Evaluate the model's performance and retrieve metrics
    precision_per_class, recall_per_class, f1_per_class, accuracy, roc_auc = model_performance(y_test, y_pred, y_pred_proba)

    # Storing the metrics values in a dictionary
    model_scores = {
        "Model": model_name,
        "Balancing": balance,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        **{f"Precision Class {i}": p for i, p in enumerate(precision_per_class)},
        **{f"Recall Class {i}": r for i, r in enumerate(recall_per_class)},
        **{f"F1 Class {i}": f for i, f in enumerate(f1_per_class)},
    }

    # Convert the dictionary to a single-row DataFrame for easy comparison across models
    return pd.DataFrame([model_scores])


def kfold(model_name: str, model, balance: str, X, y, balancing = None, weights = False):
    """
    Perform K-Fold cross-validation on a given machine learning model, evaluating its performance with different metrics.

    Parameters:
        model_name: The name of the machine learning model being evaluated.
        model: The machine learning model to be trained and evaluated.
        balance: A description of the class balancing method applied (or "None" if no balancing is applied).
        X: Features dataset.
        y: Target dataset.
        balancing (optional): A class balancing technique.
        weights (optional): Sample weights for training the model.

    Returns:
        A DataFrame with the average performance metrics for the given model across all folds.
        Includes accuracy, ROC-AUC, precision, recall, and F1 score for each class.
    """

    # Configure StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Placeholder for results from each fold
    fold_scores = []

    # Create the K-Fold splits
    for train_index, test_index in skf.split(X, y):

        # Split the data into training and testing subsets based on the current fold
        X_train_f, X_test_f = X.iloc[train_index], X.iloc[test_index]
        y_train_f, y_test_f = y.iloc[train_index], y.iloc[test_index]

        # Compute fold scores
        scores = model_scores(model_name, model, balance, X_train_f, y_train_f, X_test_f, y_test_f, balancing, weights)
        fold_scores.append(scores)
        
        # Combine all fold scores into a single DataFrame
        fold_df = pd.concat(fold_scores, ignore_index=True)

    # Calculate the mean of the numeric columns (performance metrics) across all folds
    numeric_means = fold_df.mean(numeric_only=True).to_frame().T
    numeric_means["Model"] = model_name
    numeric_means["Balancing"] = balance

    # Reorder the columns to match the original order
    columns = fold_df.columns.tolist()
    numeric_means = numeric_means[columns]

    return numeric_means