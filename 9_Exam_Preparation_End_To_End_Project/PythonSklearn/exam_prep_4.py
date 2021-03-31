import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import f1_score, classification_report, make_scorer

from sklearn.linear_model import LinearRegression, RANSACRegressor

from sklearn import datasets

# https://medium.com/berkeleyischool/how-to-use-machine-learning-to-predict-hospital-readmissions-part-1-bd137cbdba07
# https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008#
diabetes_dataset = datasets.load_diabetes()
diabetes_data = pd.DataFrame(diabetes_dataset.data)
print(diabetes_dataset.DESCR)
# See diabetes_dataset.DESCR
diabetes_data.columns = ["Age", "Sex", "BMI", "S1", "S2", "S3", "S4", "S5", "S6", "Y"]

print("diabetes_data.describe()")
print(diabetes_data.describe())

# No missing data
print("diabetes_data.info()")
print(diabetes_data.info())

diabetes_attributes = diabetes_data.drop("Y", axis=1)
diabetes_labels = diabetes_data.Y

print("initial diabetes_attributes.shape ", diabetes_attributes.shape)

scaler = MinMaxScaler()
scaler.fit(diabetes_attributes)
diabetes_attributes = scaler.transform(diabetes_attributes)

(
    diabetes_attributes_train,
    diabetes_attributes_test,
    diabetes_labels_train,
    diabetes_labels_test,
) = train_test_split(diabetes_attributes, diabetes_labels, train_size=0.8)

print("LinearRegression")
logistic_regression = LinearRegression()
logistic_regression.fit(diabetes_attributes_train, diabetes_labels_train)

# Not good score for this data set
print("Train score")
print(logistic_regression.score(diabetes_attributes_train, diabetes_labels_train))

print("Test score")
print(logistic_regression.score(diabetes_attributes_test, diabetes_labels_test))

ransac = RANSACRegressor(
    LinearRegression(), min_samples=50, max_trials=100, residual_threshold=5.0
)
ransac.fit(diabetes_attributes_train, diabetes_labels_train)

# Check the accuracy score
inliers = diabetes_attributes_train[ransac.inlier_mask_]
outliers = diabetes_attributes_train[~ransac.inlier_mask_]
print(
    "Train Inliers RANSAC Model score: ",
    ransac.score(inliers, diabetes_labels_train[ransac.inlier_mask_]),
)
# print("Train Outliers RANSAC Model score: ", ransac.score(outliers, diabetes_labels_train[~ransac.inlier_mask_]))
