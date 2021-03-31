import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

adult_income_data = pd.read_csv("../Data/adult.data")
adult_income_data.columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

print(adult_income_data.columns)
print(adult_income_data.head())

# Remove curse
adult_income_data = adult_income_data.drop("fnlwgt", axis=1)

attributes = adult_income_data.drop("income", axis=1)
labels = adult_income_data.income

attributes = pd.get_dummies(attributes)


attributes_train, attributes_test, labels_train, labels_test = train_test_split(
    attributes, labels, train_size=0.7, stratify=labels
)

tree = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=100, learning_rate=0.1)
ada.fit(attributes_train, labels_train)

predicted_labels_train = ada.predict(attributes_train)
print(f1_score(labels_train, predicted_labels_train, pos_label=" >50K"))

predicted_labels_test = ada.predict(attributes_test)
print(f1_score(labels_test, predicted_labels_test, pos_label=" >50K"))

print(classification_report(labels_train, predicted_labels_train))

print(classification_report(labels_test, predicted_labels_test))

train_pred = accuracy_score(labels_train, ada.predict(attributes_train))
test_pred = accuracy_score(labels_test, ada.predict(attributes_test))
print("AdaBoost tree train / test accuracies: {0} / {1}".format(train_pred, test_pred))
