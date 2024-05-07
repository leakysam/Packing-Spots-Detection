import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

#prepare data
input_dir = '/Volumes/mac/School/Parking spots/clf-data'
categories = ['empty','not_empty']
data = []
labels = []
for category_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))#resize all images to this 
        data.append(img.flatten())#take the image in an array
        labels.append(category_index)
#cast the list into numpy array
data = np.asarray(data)
labels = np.asarray(labels)        
    
#train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)# 20% to test, shuffle to avoid bias when training, stratify to group them into categories and ensure its portioned to represent the whole data

#train classifier
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}] #train many combinations for keys gamma and c[12].  image classifiers look at sk SVC documentation choose the best
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)# train all 12 image classifiers

#test perfomance
best_estimator = grid_search.best_estimator_   #get the best of all image classifiers from the 12
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified'.format(str(score * 100)))
#save the model to be used later
pickle.dump(best_estimator, open('./model.p', 'wb'))