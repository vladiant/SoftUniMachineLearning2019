import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# housing_data = pd.read_fwf("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None)
housing_data = pd.read_fwf("../Data/housing.data", header=None)
housing_data.columns = [
    "crime_rate",
    "zoned_land",
    "industry",
    "bounds_river",
    "nox_conc",
    "rooms",
    "age",
    "distance",
    "highways",
    "tax",
    "pt_ratio",
    "b_estimator",
    "pop_stat",
    "price",
]

housing_data_input = housing_data.drop("price", axis=1)
housing_data_output = housing_data["price"]

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

"""
for column_label in housing_data_input.columns:
    # print(column_label)
    plt.title(column_label)
    plt.scatter(housing_data_input[column_label], housing_data_output, label = "original data")
    plt.scatter(housing_data_input[column_label], model.predict(housing_data_input), label = "fitted data")
    # plt.scatter((min_x, min_y), (max_x, max_y), label = "fitted data")
    plt.legend()
    plt.show()
"""

polynomial_features = PolynomialFeatures(2, interaction_only=False)
polynomial_input = polynomial_features.fit_transform(housing_data_input)

# Check shape - there should be less features than measurements
print(polynomial_input.shape)

polynomial_model = LinearRegression()
polynomial_model.fit(polynomial_input, housing_data_output)

# Check the accuracy score
print(
    "Polynomial model score:",
    polynomial_model.score(polynomial_input, housing_data_output),
)

"""
for column_label in range(polynomial_input.shape[1]):
    # print(column_label)
    plt.title(column_label)
    plt.scatter(polynomial_input[:, column_label], housing_data_output, label = "original data")
    plt.scatter(polynomial_input[:, column_label], polynomial_model.predict(polynomial_input), label = "fitted data")
    # plt.scatter((min_x, min_y), (max_x, max_y), label = "fitted data")
    plt.legend()
    plt.show()
"""
