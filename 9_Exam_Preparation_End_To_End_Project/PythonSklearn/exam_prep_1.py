import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import f1_score, classification_report, make_scorer

from sklearn.linear_model import LogisticRegression

# https://medium.com/berkeleyischool/how-to-use-machine-learning-to-predict-hospital-readmissions-part-1-bd137cbdba07
# https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008#
diabetes_data = pd.read_csv("../Data/diabetic_data.zip", true_values=['YES'], false_values=['NO'], na_values=['?'])

print(diabetes_data.head())

print("diabetes_data.shape ", diabetes_data.shape)

print("diabetes_data.encounter_id.unique() ", diabetes_data.encounter_id.unique())

print('diabetes_data.groupby("encounter_id").size()', diabetes_data.groupby("encounter_id").size())

print("diabetes_data.describe()")
print(diabetes_data.describe())

# No missing data
print("diabetes_data.info()")
print(diabetes_data.info())

print('diabetes_data.groupby("readmitted").size() ', diabetes_data.groupby("readmitted").size())

print("diabetes_data.discharge_disposition_id.unique() ", diabetes_data.discharge_disposition_id.unique())

print("diabetes_data.admission_source_id.unique() ", diabetes_data.admission_source_id.unique())

print("---")
print('pd.crosstab(diabetes_data.admission_source_id, diabetes_data.readmitted)')
print(diabetes_data.groupby("admission_source_id").size() * 100 / len(diabetes_data))
print("---")

print("diabetes_data.admission_source_id.unique() ",
      pd.crosstab(diabetes_data.admission_source_id, diabetes_data.readmitted))

print("pd.crosstab(diabetes_data.admission_source_id, diabetes_data.readmitted) ",
      pd.crosstab(diabetes_data.admission_source_id, diabetes_data.readmitted))

# Remove insignificant columns
diabetes_data = diabetes_data.drop(["encounter_id", "patient_nbr", "payer_code"], axis=1)

print("diabetes_data.age.unique() ", diabetes_data.age.unique())

print("diabetes_data.weight.unique() ", diabetes_data.weight.unique())

print("diabetes_data.insulin.unique() ", diabetes_data.insulin.unique())

diabetes_attributes = diabetes_data.drop("readmitted", axis=1)
diabetes_labels = diabetes_data.readmitted

print("initial diabetes_attributes.shape ", diabetes_attributes.shape)

diabetes_attributes = pd.get_dummies(diabetes_attributes)

print("get_dummies diabetes_attributes.shape ", diabetes_attributes.shape)

# print(np.histogram(diabetes_attributes.num_lab_procedures, bins='fd'))

scaler = MinMaxScaler()
scaler.fit(diabetes_attributes)
diabetes_attributes = scaler.transform(diabetes_attributes)

diabetes_attributes_train, diabetes_attributes_test, diabetes_labels_train, diabetes_labels_test \
    = train_test_split(diabetes_attributes, diabetes_labels, train_size=0.9, stratify=diabetes_labels)

print("LogisticRegression")
logistic_regression = LogisticRegression()
logistic_regression.fit(diabetes_attributes_train, diabetes_labels_train)

# Not good score for this data set
print("Train score")
print(logistic_regression.score(diabetes_attributes_train, diabetes_labels_train))

# f1 score is useful for one label only
train_predictions_log_regr = logistic_regression.predict(diabetes_attributes_train)
print("Train data")
print(classification_report(diabetes_labels_train, train_predictions_log_regr))

test_predictions_log_regr = logistic_regression.predict(diabetes_attributes_test)
print("Test data")
print(classification_report(diabetes_labels_test, test_predictions_log_regr))

'''
train
              precision    recall  f1-score   support

         <30       0.55      0.02      0.04     10221
         >30       0.53      0.35      0.42     31990
          NO       0.61      0.87      0.72     49378

    accuracy                           0.59     91589
   macro avg       0.56      0.41      0.39     91589
weighted avg       0.58      0.59      0.54     91589


test
              precision    recall  f1-score   support

         <30       0.39      0.02      0.03      1136
         >30       0.50      0.34      0.40      3555
          NO       0.60      0.85      0.71      5486

    accuracy                           0.58     10177
   macro avg       0.50      0.40      0.38     10177
weighted avg       0.54      0.58      0.52     10177
'''

# low f1_score - maybe too different persons -> oversampling
# high bias!!!
# reduce regularization ?

print("GridSearchCV + LogisticRegression")

params = {
    "C": [100, 1000, 1e5],
}
# pos_label=">30" in make_scorer initial approach
grid_search_log_regr = GridSearchCV(LogisticRegression(), params, scoring=make_scorer(f1_score, average='weighted'))
grid_search_log_regr.fit(diabetes_attributes_train, diabetes_labels_train)

print("best_score: ", grid_search_log_regr.best_score_)

print("best_params: ", grid_search_log_regr.best_params_)

best_model = grid_search_log_regr.best_estimator_

train_predictions_log_regr = best_model.predict(diabetes_attributes_train)
print("Train data")
print(classification_report(diabetes_labels_train, train_predictions_log_regr))

test_predictions_log_regr = best_model.predict(diabetes_attributes_test)
print("Test data")
print(classification_report(diabetes_labels_test, test_predictions_log_regr))
