import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

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

print("AdaBoostClassifier model fit")
ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200)
ada_boost.fit(diabetes_attributes_train, diabetes_labels_train)

train_predictions_log_regr = ada_boost.predict(diabetes_attributes_train)
print("Train data")
print(classification_report(diabetes_labels_train, train_predictions_log_regr))

test_predictions_log_regr = ada_boost.predict(diabetes_attributes_test)
print("Test data")
print(classification_report(diabetes_labels_test, test_predictions_log_regr))

"""
n_estimators=50

              precision    recall  f1-score   support

         <30       0.48      0.06      0.11     10221
         >30       0.51      0.40      0.45     31990
          NO       0.63      0.83      0.71     49378

    accuracy                           0.59     91589
   macro avg       0.54      0.43      0.42     91589
weighted avg       0.57      0.59      0.55     91589

              precision    recall  f1-score   support

         <30       0.39      0.05      0.09      1136
         >30       0.51      0.40      0.45      3555
          NO       0.62      0.82      0.71      5486

    accuracy                           0.59     10177
   macro avg       0.51      0.42      0.42     10177
weighted avg       0.56      0.59      0.55     10177
"""
