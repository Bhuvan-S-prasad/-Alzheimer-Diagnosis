# Alzheimer's Disease Prediction Project

## Table of Contents

1.  [General Overview](#general-overview)
2.  [Exploratory Data Analysis (EDA) Findings](#exploratory-data-analysis-eda-findings)
3.  [Models Used and Model Building](#models-used-and-model-building)
4.  [MLOps Integration](#mlops-integration)
5.  [Project Outcomes and Functionality](#project-outcomes-and-functionality)
6.  [Project Directory Structure](#project-directory-structure)
7.  [Conclusion](#conclusion)
8.  [Future Work](#future-work)
9.  [License](#license)

## 1. General Overview

📌 This project focuses on developing an machine learning application for the early prediction of Alzheimer's disease.  Alzheimer's is a devastating neurodegenerative disease, and early detection can significantly improve patient outcomes and allow for timely interventions.  This project addresses this critical need by building a binary classification model to predict the likelihood of an Alzheimer's diagnosis based on a dataset containing various patient features related to demographics, lifestyle, health history, and cognitive assessments. We employed a range of machine learning models, including Logistic Regression, Random Forest,  to tackle this classification task.  Crucially, this project emphasizes robust software engineering practices by integrating Machine Learning Operations (MLOps) principles throughout the development lifecycle. This includes automated data preprocessing and model training pipelines, version control for code and configurations, artifact management for models and data, and considerations for reproducibility and deployment, aiming to create a reliable and maintainable machine learning solution. We utilized a CSV dataset containing  records 74,284 and 24 features, with "Alzheimer's Diagnosis" as the target variable.


## 2. Exploratory Data Analysis (EDA) Findings

Exploratory Data Analysis (EDA) aimed to understand the characteristics of the Alzheimer's dataset and identify potential patterns and relationships within the features and the target variable "Alzheimer's Diagnosis". Key findings from the EDA are summarized below:

- **Target Variable Distribution ("Alzheimer's Diagnosis"):**
    *   The target variable, indicating Alzheimer's diagnosis (Yes/No), exhibits a class imbalance. The "No" class is significantly more frequent than the "Yes" class, representing approximately No: 58% and Yes: 42%, respectively.

- **Numerical Features:**
    *   **Age:** The distribution of age is U-shaped, with peaks at 50 years and 94 years ages and lower frequency in middle age ranges. Box plots revealed that the "Yes" Alzheimer's group tends to have a noticeably higher median age compared to the "No" group, and correlation analysis showed a moderate positive correlation 0.42 with Alzheimer's diagnosis, indicating age is a potentially important predictor.
    *   **Cognitive Test Score:** The distribution of Cognitive Test Score is plateau-like. Box plots showed that individuals with an Alzheimer's diagnosis "Yes" group tend to have lower Cognitive Test Scores on average compared to the "No" group. Despite this visual difference and clinical relevance, the linear correlation with Alzheimer's diagnosis was near zero, suggesting a potentially non-linear relationship.
    *   **BMI (Body Mass Index):** The distribution of BMI is relatively uniform. Box plots showed very little visual difference in BMI distribution between the "Yes" and "No" Alzheimer's groups, and correlation analysis confirmed a near-zero linear correlation with the target variable, suggesting BMI might be less directly informative for prediction in this dataset.
    *   **Education Level:** The distribution of Education Level is discrete and somewhat uniform. Box plots indicated a subtle trend towards slightly lower median Education Levels in the "Yes" Alzheimer's group compared to the "No" group. However, linear correlation with Alzheimer's diagnosis was negligible.

*   **Correlation Matrix of Numerical Features:**
    *   The correlation matrix of numerical features Age, Education Level, BMI, Cognitive Test Score are very weak linear correlations between these features themselves, showing small multicollinearity among this set of numerical predictors.
    *   Age showed a moderate positive linear correlation with Alzheimer's Diagnosis 0.42, while Education Level, BMI, and Cognitive Test Score exhibited near-zero linear correlations with the target.



## 3. Models Used and Model Building


this project is done using two machine learning models: Logistic Regression and Random Forest. These models were chosen as they represent a good starting point for binary classification problems, with Logistic Regression serving as an interpretable baseline and Random Forest capable of capturing more complex, non-linear relationships in the data.

*   **Logistic Regression:** A linear model that predicts the probability of a binary outcome. It is known for its interpretability and efficiency, making it a useful baseline model.
*   **Random Forest:** An ensemble learning method that builds multiple decision trees and aggregates their predictions. Random Forests are robust, often perform well on complex datasets, and can capture non-linear relationships and feature interactions.

**Data Preprocessing Pipeline:**

Before training the models,  data preprocessing pipeline is done in  the `src/preprocess_data.ipynb` to prepare the dataset for machine learning. The steps in this pipeline were:

1.  **Target Variable Encoding:** The categorical "Alzheimer's Diagnosis" target variable ("Yes"/"No") was converted into numerical format (1 for "Yes", 0 for "No") to be compatible with machine learning algorithms.
2.  **Categorical Feature Encoding:**  Categorical features in the dataset were transformed into numerical representations using `One-Hot Encoding`. This converts each categorical feature into multiple binary features, representing the presence or absence of each category. The `OneHotEncoder` from scikit-learn was used with `handle_unknown='ignore'` to handle potential unseen categories during prediction.
3.  **Numerical Feature Scaling:** Numerical features Age, Education Level, BMI, Cognitive Test Score were standardized using `StandardScaler` from scikit-learn. This scales numerical features to have zero mean and unit variance, which can improve the performance of some machine learning algorithms.
4.  **Data Splitting:** The preprocessed data was split into three sets:
    *   **Training Set (approximately 70%):** Used to train the machine learning models.
    *   **Validation Set (approximately 15%):** Used for model evaluation during training and for model selection.
    *   **Test Set (approximately 15%):**  Held-out dataset used for final, unbiased evaluation of the trained models' performance.
    Stratified splitting was employed to maintain the class distribution of the target variable across all sets, addressing the class imbalance observed in EDA.

The fitted preprocessing pipeline and the processed datasets training, validation, and test sets were saved as artifacts in the `models/` and `data/processed/` directories, respectively, for reproducibility and later use in the prediction application.

**Model Training:**

Model training was automated using the `src/train_model.ipynb` script.  For both Logistic Regression and Random Forest models:

*   Models were trained using the preprocessed training data.
*   Hyperparameter tuning was performed using GridSearchCV with cross-validation on the training data, evaluated on the validation set. tuning key hyperparameters for both Logistic Regression (regularization strength 'C') and Random Forest (number of estimators 'n_estimators', maximum depth 'max_depth') to optimize model performance.
*   The primary scoring metric used for hyperparameter tuning was ROC AUC (Area Under the Receiver Operating Characteristic curve), which is suitable for imbalanced datasets and focuses on ranking performance.

**Model Evaluation:**

After training and hyperparameter tuning, the performance of both Logistic Regression and Random Forest models was evaluated on both the validation and the held-out test sets. We used a range of classification metrics to comprehensively assess model performance, including:

*   **Accuracy:** Overall correctness of predictions.
*   **Precision:** Proportion of correctly predicted "Yes" cases out of all predicted "Yes" cases.
*   **Recall:** Proportion of correctly predicted "Yes" cases out of all actual "Yes" cases.
*   **F1-Score:** Harmonic mean of precision and recall, balancing both metrics.
*   **AUC-ROC (Area Under the ROC Curve):** Measures the model's ability to distinguish between classes across different probability thresholds.
*   **Classification Report:** Provides a detailed breakdown of precision, recall, F1-score, and support for each class.
*   **ROC Curve Visualization:** ROC curves were generated and saved in the `reports/` directory to visually assess the trade-off between true positive rate and false positive rate for each model.



## 4. MLOps Integration


This project incorporates several Machine Learning Operations (MLOps) principles to enhance its reliability, reproducibility, and maintainability. Key MLOps practices implemented include:

*   **Version Control (Git):** using Git for comprehensive version control, tracking all changes to the codebase python and ipynb scripts, configuration files `config.yaml`, and documentation `README.md`. This ensures a complete history of project modifications, facilitates collaboration, and enables easy rollback to previous states if needed.useing GitHub to host the project repository, further enhancing collaboration and code management.

*   **Configuration Management (YAML):** Project configurations, such as data paths, feature lists, model hyperparameters, and training settings, are managed through a centralized `config.yaml` file. This externalizes configuration from the code, making the project more flexible, adaptable to different environments, and easier to maintain. The [YAML](https://yaml.org/) format was chosen for its human-readability and ease of use in configuration files.

*   **Automated Pipelines (Python Scripts):** We implemented automated pipelines using Python scripts `src/preprocess_data.ipynb` and `src/train_model.ipynb` for critical stages of the ML lifecycle:
    *   **Data Preprocessing Pipeline - `preprocess_data.py`:** Automates data loading, cleaning, preprocessing (feature encoding, scaling), and splitting into training, validation, and test sets. This ensures consistent data preparation and eliminates manual, error-prone steps.
    *   **Model Training Pipeline - `train_model.py`:** Automates the process of training machine learning models, hyperparameter tuning, evaluation, and saving trained models and evaluation metrics. This streamlines experimentation, ensures reproducibility of training runs, and facilitates model management.

*   **Artifact Management - Directory Structure and File Saving:** using structured project directory to organize and manage various project artifacts:
    *   `data/`: Stores raw and processed datasets, ensuring clear separation and versioning of data.
    *   `models/`:  Dedicated directory for saving trained machine learning models, preprocessing objects (`preprocessor.joblib`), and feature name lists (`feature_names.joblib`). This centralizes model artifacts for easy access and deployment.
    *   `reports/`: Stores evaluation metrics in `model_metrics.json` and visualizations e.g., ROC curves, providing organized access to experiment results.
      using  joblib` library for efficient saving and loading of scikit-learn objects (preprocessor, models) and standard file saving methods.

*   **Reproducibility (Virtual Environments and `requirements.txt`):** To ensure reproducibility of the project environment, we utilized Python virtual environments to isolate project dependencies. A `requirements.txt` file is included to list all Python package dependencies and their versions, allowing others (or your future self) to easily recreate the exact project environment and reproduce the project's results.





## 6. Project Directory Structure
```
alzheimer_prediction_project/
├── data/
│ ├── raw/
│ │ └── your_dataset.csv
│ └── processed/
│ ├── X_train_processed.csv
│ ├── X_val_processed.csv
│ ├── X_test_processed.csv
│ ├── y_train.csv
│ ├── y_val.csv
│ └── y_test.csv
├── models/
│ ├── preprocessor.joblib
│ ├── trained_models.joblib
│ └── feature_names.joblib
├── reports/
│ ├── model_metrics.json
│ └── *_roc_curve.png
├── src/
│ ├── preprocess_data.py
│ └── train_model.py
│ └── predict.py
├── config.yaml
├── README.md
└── requirements.txt
```

## 7. Conclusion

This project explored a machine learning approach for Alzheimer’s disease prediction, covering data exploration, preprocessing, model training, evaluation, and deployment as a simple command-line tool.
Since this was my first experience with MLOps, I focused on implementing basic MLOps concepts to enhance the project's organization, reproducibility, and maintainability. This included automated pipelines, version control, and experiment tracking—keeping things simple yet effective.
Overall, this project provided valuable hands-on experience with MLOps fundamentals and laid a strong foundation for tackling more advanced ML projects in the future
