import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt

preset_iris_data = datasets.load_iris()

# print(preset_iris_data.data)
# print(preset_iris_data.target)
# print(preset_iris_data.target_names)
# print(preset_iris_data.DESCR)

iris_data = pd.DataFrame(preset_iris_data.data)

# See preset_iris_data.DESCR
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

iris_data['class'] = pd.DataFrame(preset_iris_data.target)

print(iris_data.head())

# Check the data columns are OK
print(iris_data.shape)
# Prints the number of data for each class
# To check that the data is balanced
print(iris_data.groupby('class').size())

for class_name, data in iris_data.groupby('class'):
    # plt.scatter(data.petal_width, data.petal_length, label=class_name)
    plt.scatter(data.petal_width, data.petal_length, label=preset_iris_data.target_names[class_name])

plt.legend()
plt.show()

for class_name, data in iris_data.groupby('class'):
    # plt.scatter(data.sepal_width, data.sepal_length, label=class_name)
    plt.scatter(data.sepal_width, data.sepal_length, label=preset_iris_data.target_names[class_name])

plt.legend()
plt.show()

iris_model = LogisticRegression(C=1e9)

iris_data_input = iris_data.drop("class", axis=1)
iris_data_output = iris_data["class"]

# Check shapes of the data
print(iris_data_input.shape)
print(iris_data_output.shape)

iris_model.fit(iris_data_input, iris_data_output)

# Check the accuracy score
print(iris_model.score(iris_data_input, iris_data_output))

# Check prediction for some data
print([(preset_iris_data.target_names[x]) for x in iris_model.predict(iris_data_input[:10])])

# Get random sample
testing_sample = iris_data.sample(10)
testing_sample_input = testing_sample.drop("class", axis=1)
testing_sample_output = testing_sample["class"]

testing_sample_predict = iris_model.predict(testing_sample_input)
print(testing_sample_predict)

for predicted, expected in zip(testing_sample_predict, testing_sample_output):
    print(predicted, expected)
    if predicted != expected:
        print("Expected: ", preset_iris_data.target_names[expected], "  Predicted: ",
              preset_iris_data.target_names[predicted])

# Linear coefficient
print(iris_model.coef_)

# Intercept
print(iris_model.intercept_)
