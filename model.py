import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# import dataset
df = pd.read_csv('Iris.csv')
print(df)

y = df["Species"]
x = df.drop(['Id','Species'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Make pickle
pickle.dump(model, open("model.pkl", "wb"))

# print(x)