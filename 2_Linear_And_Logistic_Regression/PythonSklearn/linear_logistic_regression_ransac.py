import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# housing_data = pd.read_fwf("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None)
housing_data = pd.read_fwf("../Data/housing.data", header=None)
housing_data.columns = ["crime_rate", "zoned_land", "industry", "bounds_river", "nox_conc", "rooms", "age", "distance",
                        "highways", "tax", "pt_ratio", "b_estimator", "pop_stat", "price"]

# Check data
print(housing_data.head())

housing_data_input = housing_data.drop("price", axis=1)
housing_data_output = housing_data["price"]

# Check data
print(housing_data_input.head())
print(housing_data_output.head())

# Check shapes of the data
print(housing_data_input.shape)
print(housing_data_output.shape)

# X will be copied
# Intercept will be calculated
# No jobs to be run in parallel
model = LinearRegression()
model.fit(housing_data_input, housing_data_output)

# Check the accuracy score
print("Model score: ", model.score(housing_data_input, housing_data_output))

# Linear coefficient
# print(model.coef_)

# Intercept
# print(model.intercept_)

# Actual lost function for a sample
# predicted_values = model.predict(housing_data_input[:10])
# actual_values = housing_data_output[:10]
# print(np.sqrt((predicted_values -actual_values) ** 2))

ransac = RANSACRegressor(LinearRegression(), min_samples=50, max_trials=100, residual_threshold=5.0)
ransac.fit(housing_data_input, housing_data_output)

# Check the accuracy score
inliers = housing_data_input[ransac.inlier_mask_]
outliers = housing_data_input[~ransac.inlier_mask_]
print("Inliers RANSAC Model score: ", ransac.score(inliers, housing_data_output[ransac.inlier_mask_]))
print("Outliers RANSAC Model score: ", ransac.score(outliers, housing_data_output[~ransac.inlier_mask_]))

for column_label in housing_data_input.columns:
    # print(column_label)
    plt.title(column_label)
    plt.scatter(inliers[column_label], housing_data_output[ransac.inlier_mask_], label = "inlier original data")
    # plt.scatter(inliers[column_label], model.predict(inliers), label = "inlier fitted data")
    plt.scatter(outliers[column_label], housing_data_output[~ransac.inlier_mask_], label = "outlier original data")
    # plt.scatter(outliers[column_label], model.predict(outliers), label = "outlier fitted data")
    plt.legend()
    plt.show()



