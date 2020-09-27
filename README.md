# Supervised_Machine_Learning

# Evaluating Credit Risk
The growth of personal lending in 2019 has many lending firms using machine learning techniques to evaluate and predict credit risk.  Here, we will use four machine learning algorithms for an analysis of a dataset with over 68,000 rows of loan data. For each model, we resample the data to address class imbalance (low- vs. high-risk) and train a logistic regression classifier (from Scikit-learn) using the resampled data. Then we calculate a balanced accuracy score using balanced_accuracy_score from sklearn.metrics. A confusion_matrix is generated for each model and a classification report (classification_report_imbalanced from imblearn.metrics) is created.

## Testing Results From the Four Models

### Balanced Accuracy Scores
- 0.6495 -> Naive Random Oversampling
- 0.6585 -> SMOTE Oversampling
- 0.5443 -> Cluster Centroids Undersampling   
- 0.6622 -> SMOTEEN Combo Over/Under Sampling 

The accuracy score shows the accuracy of correct predictions.  The Cluster Centroid has the worst score. The other models, while similar in score, show the SMOTEEN model to be slightly superior.

### Precision
Precision is the measure of reliabilty of a positive classifcation. 
Precision = TruePos/(TruePos + FalsePos)
All four models show the low-risk loans to be predicted accurately at 1.00. 
The high-risk loans, however, show a precision of 0.01 which means many of the high-risk loans are improperly catgorized.


### Recall/Sensitivity
Sensitivity of the model shows how likely it is to be correctly predicted positive.
Sensitivity = TruePos/(TruePos + FalseNeg)
- high-risk=0.73 /low-risk=0.57 -> Naive Random Oversampling
- high-risk=0.63 /low-risk=0.68 -> SMOTE Oversampling
- high-risk=0.67 /low-risk=0.42 -> low-risk= Centroids Undersampling   
- high-risk=0.78 /low-risk=0.54 -> SMOTEEN Combo Over/Under Sampling 

Again, the SMOTEEN model is superior to the other three models.

## Recommendation
If these were the only four models we could use, the best choice would be the SMOTEEN Over/Under Sampling as it had the best scores. 

## Resources
- Python 3.7 in Jupyter Notebook
- Libraries: pandas, numpy, path, counter
  - scikit-learn: sklearn.linear_model, sklearn.metrics, sklearn.model_selection
  - imbalanced-learn: imblearn.over_sampling, imblearn.metrics, imblearn.over_sampling, imblearn.under_sampling, imblearn.combine
- Data: LoanStats_2019Q1.csv from LendingClub