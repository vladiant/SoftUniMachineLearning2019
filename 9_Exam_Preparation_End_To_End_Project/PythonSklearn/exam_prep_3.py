import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

# https://medium.com/berkeleyischool/how-to-use-machine-learning-to-predict-hospital-readmissions-part-1-bd137cbdba07
# https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008#
diabetes_data = pd.read_csv(
    "../Data/diabetic_data.zip",
    true_values=["YES"],
    false_values=["NO"],
    na_values=["?"],
)

# Remove insignificant columns
diabetes_data = diabetes_data.drop(
    ["encounter_id", "patient_nbr", "payer_code"], axis=1
)

diabetes_attributes = diabetes_data.drop("readmitted", axis=1)
diabetes_labels = diabetes_data.readmitted

diabetes_attributes = pd.get_dummies(diabetes_attributes)

# print(np.histogram(diabetes_attributes.num_lab_procedures, bins='fd'))

scaler = MinMaxScaler()
scaler.fit(diabetes_attributes)
diabetes_attributes = scaler.transform(diabetes_attributes)

(
    diabetes_attributes_train,
    diabetes_attributes_test,
    diabetes_labels_train,
    diabetes_labels_test,
) = train_test_split(
    diabetes_attributes, diabetes_labels, train_size=0.9, stratify=diabetes_labels
)

# PCA to reduce dimensionality
print("PCA calculation")

pca = PCA(n_components=300)
pca.fit(diabetes_attributes_train)

# Did it worked?
print("pca.explained_variance_ratio_ ", pca.explained_variance_ratio_)

plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.show()

print(pca.explained_variance_ratio_.sum())

total_variance = np.cumsum(pca.explained_variance_ratio_)

plt.bar(range(len(total_variance)), total_variance)
# plt.ylim(0, 0.025)
plt.show()

# Part of data to 90%
print("Data < 90% ", len(total_variance[total_variance < 0.9]))

diabetes_attributes_pca = pca.transform(diabetes_attributes)

(
    diabetes_attributes_train,
    diabetes_attributes_test,
    diabetes_labels_train,
    diabetes_labels_test,
) = train_test_split(
    diabetes_attributes_pca, diabetes_labels, train_size=0.9, stratify=diabetes_labels
)

logistic_regressions = [
    LogisticRegression(C=100),
    LogisticRegression(C=1e5),
    LogisticRegression(C=1e7),
]

for model in logistic_regressions:
    print("Logistic regression model C= ", model.C)
    # Use PCA data here!
    model.fit(diabetes_attributes_train, diabetes_labels_train)

    train_predictions_log_regr = model.predict(diabetes_attributes_train)
    print("Train results")
    print(classification_report(diabetes_labels_train, train_predictions_log_regr))

    test_predictions_log_regr = model.predict(diabetes_attributes_test)
    print("Test results")
    print(classification_report(diabetes_labels_test, test_predictions_log_regr))

"""
LogisticRegression(C=100):
Train results
              precision    recall  f1-score   support

         <30       0.47      0.01      0.02     10221
         >30       0.51      0.31      0.38     31990
          NO       0.60      0.87      0.71     49378

    accuracy                           0.58     91589
   macro avg       0.53      0.40      0.37     91589
weighted avg       0.55      0.58      0.52     91589

Test results
              precision    recall  f1-score   support

         <30       0.55      0.02      0.03      1136
         >30       0.51      0.31      0.38      3555
          NO       0.60      0.87      0.71      5486

    accuracy                           0.58     10177
   macro avg       0.55      0.40      0.37     10177
weighted avg       0.56      0.58      0.52     10177


LogisticRegression(C=1e5):
Train results
              precision    recall  f1-score   support

         <30       0.47      0.01      0.02     10221
         >30       0.51      0.31      0.38     31990
          NO       0.60      0.87      0.71     49378

    accuracy                           0.58     91589
   macro avg       0.53      0.40      0.37     91589
weighted avg       0.55      0.58      0.52     91589

Test results
              precision    recall  f1-score   support

         <30       0.55      0.02      0.03      1136
         >30       0.51      0.31      0.38      3555
          NO       0.60      0.87      0.71      5486

    accuracy                           0.58     10177
   macro avg       0.55      0.40      0.37     10177
weighted avg       0.56      0.58      0.52     10177


LogisticRegression(C=1e7):
Train results
              precision    recall  f1-score   support

         <30       0.47      0.01      0.02     10221
         >30       0.51      0.31      0.38     31990
          NO       0.60      0.87      0.71     49378

    accuracy                           0.58     91589
   macro avg       0.53      0.40      0.37     91589
weighted avg       0.55      0.58      0.52     91589

Test results
              precision    recall  f1-score   support

         <30       0.55      0.02      0.03      1136
         >30       0.51      0.31      0.38      3555
          NO       0.60      0.87      0.71      5486

    accuracy                           0.58     10177
   macro avg       0.55      0.40      0.37     10177
weighted avg       0.56      0.58      0.52     10177
"""
