import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare data
input_dir = '/Volumes/mac/School/Parking spots/clf-data'
categories = ['empty', 'not_empty']
data = []
labels = []

for category_index, category in enumerate(categories):
    category_folder = os.path.join(input_dir, category)
    for file in os.listdir(category_folder):
        img_path = os.path.join(category_folder, file)
        img = imread(img_path)
        img = resize(img, (15, 15))  # Resize all images to 15x15
        data.append(img.flatten())
        labels.append(category_index)

data = np.array(data)
labels = np.array(labels)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train classifier using Grid Search
classifier = SVC()
parameters = {'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# Evaluate the model
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_test, y_prediction)
print(f"{score * 100:.2f}% of samples were correctly classified.")

# Confusion matrix
cm = confusion_matrix(y_test, y_prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model
pickle.dump(best_estimator, open('./model.p', 'wb'))
