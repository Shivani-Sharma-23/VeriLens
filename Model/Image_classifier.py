import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Preparing Data
input_dir ="C:\\Users\\shiva\\Documents\\AI_Model\\Model\\archive"
categories = ['training_fake','training_real']

data = []
labels =[]

for category_idx,category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

#train/test split
x_train, x_test , y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle =True, stratify = labels)

#TRAINING CLASSIFIER

#SVM
classifier =  SVC()

parameters = [{'gamma':[0.01, 0.001, 0.0001 ], 'C':[1,10,100,1000]}]

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
svm_accuracy = accuracy_score(y_test, y_prediction)
print("SVM Accuracy:", svm_accuracy)
svm_classification_report = classification_report(y_test,y_prediction)
print("\nSVM Classification Report:\n", svm_classification_report)


#KNN
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
knn_predictions = knn_model.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)
knn_classification_report = classification_report(y_test, knn_predictions)
print("\nKNN Classification Report:\n", knn_classification_report)


#DECISION TREE
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("\nDecision Tree Accuracy:", dt_accuracy)
dt_classification_report = classification_report(y_test, dt_predictions)
print("Decision Tree Classification Report:\n", dt_classification_report)


pickle.dump((svm_classification_report,knn_classification_report,dt_classification_report,best_estimator, knn_model, dt_model), open('./model.pkl','wb'))
