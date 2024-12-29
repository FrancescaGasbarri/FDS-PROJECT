# The Influence of Class Imbalance on Classification Task

### **Overview**

Our goal was to study the **influence of class imbalance** on the performance of 5 models in **classifying celestial objects** into:
- **Galaxy**
- **Star**
- **Quasar**

The dataset used for this study was obtained from [Kaggle](https://www.kaggle.com/datasets/diraf0/sloan-digital-sky-survey-dr18/data "View the SDSS dataset on Kaggle"), made up of 100k rows and 43 columns.

### **Models**

The following models were used for classification:
- **Logistic Regression**
- **KNN**
- **Random Forest**
- **Naive Bayes**
- **XGBoost**

### **Balancing Methods**

For each model, the following class balancing methods were explored:
- **Model's baseline configuration**
- **Model's built-in mechanisms**
- **SMOTE**
- **KMeans SMOTE**
- **Random UnderSampler**
- **Cluster Centroids**

### **Process**

The entire process was repeated for:
- The **original dataset**
- Dataset **without Galaxies**
- Dataset **without Stars**
- Dataset **without Quasars**

Each combination of model and class balancing method was tested using **3-Fold Cross Validation**, and the results for each evaluation metric were averaged.

### **Evaluation Metrics**

The performance of the models was evaluated using the following metrics:
-	**Accuracy**
-	**ROC-AUC Score**
-	**Precision**
-	**Recall**
-	**F1 Score**

### **Project Structure**

The project consists of the following files:
-	*`final_proj.ipynb`*: Contains the project’s code in a Jupyter Notebook format.
-	*`functions/`*: Folder containing helper scripts:
    -	*`pipeline.py`*: Functions to compute the classification, evaluate it with different metrics, and perform 3-Fold Cross Validation.
    -	*`plots.py`*: Functions to plot the class distribution and model performance metrics as a function of balancing methods.
