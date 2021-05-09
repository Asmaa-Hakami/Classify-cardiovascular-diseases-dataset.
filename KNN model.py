import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score



#Load the dataset from CSV file.
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, 12].values

#Divide the dataset into training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


#Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Train the KNN algorithm for k = 1.
classifier_1 = KNeighborsClassifier(n_neighbors=1)
classifier_1.fit(X_train, y_train)
y_pred_1 = classifier_1.predict(X_test)

#Print its accuracy, precision, and recall.
accuracy_1 = accuracy_score(y_test, y_pred_1)
print("Accuracy of k=1:", accuracy_1)
print(classification_report(y_test, y_pred_1))
print("********************************************************************\n")


#Train the KNN algorithm for k = 3.
classifier_3 = KNeighborsClassifier(n_neighbors=3)
classifier_3.fit(X_train, y_train)
y_pred_3 = classifier_3.predict(X_test)

#Print its accuracy, precision, and recall.
accuracy_3 = accuracy_score(y_test, y_pred_3)
print("Accuracy of k=3:", accuracy_3)
print(classification_report(y_test, y_pred_3))
print("********************************************************************\n")


#Train the KNN algorithm for k = 5.
classifier_5 = KNeighborsClassifier(n_neighbors=5)
classifier_5.fit(X_train, y_train)
y_pred_5 = classifier_5.predict(X_test)

#Print its accuracy, precision, and recall.
accuracy_5 = accuracy_score(y_test, y_pred_5)
print("Accuracy of k=5:", accuracy_5)
print(classification_report(y_test, y_pred_5))
print("********************************************************************\n")


#Train the KNN algorithm for k = 10.
classifier_10 = KNeighborsClassifier(n_neighbors=10)
classifier_10.fit(X_train, y_train)
y_pred_10 = classifier_10.predict(X_test)

#Print its accuracy, precision, and recall.
print ()
accuracy_10 = accuracy_score(y_test, y_pred_10)
print("Accuracy of k=10:", accuracy_10)
print(classification_report(y_test, y_pred_10))
print("********************************************************************\n")


#Print the highest K
highest= max(accuracy_1, accuracy_3, accuracy_5, accuracy_10)
if highest == accuracy_10:
    k = "10"
elif highest == accuracy_5:
    k = "5"
elif highest == accuracy_3:
    k = "3"
else:
    k = "1"
    
print("The highest accuracy is ", highest, "for K = ",k)



