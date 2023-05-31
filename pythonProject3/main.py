import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

participants = ['6', '14', '8.4']
conditions = ['with phone', 'without phone']
labels = []

X = []
y = []

for participant in participants:
    for condition in conditions:
        file_path = f"C:/Users/rawna/Desktop/project/signals/{participant} {condition} final 12.set"
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw_crop = raw.crop(tmin=0, tmax=100)
        data = raw_crop.get_data()

        X.append(data)
        y.extend([condition] * data.shape[1])

X = np.concatenate(X, axis=1).T
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = svm.SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
