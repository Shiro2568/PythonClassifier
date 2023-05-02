from sklearn import tree
from PIL import Image
import cv2 as cv
import numpy
import os 
import itertools

files = os.listdir('./images')

#print(files)
# Labels for the features: 0 for apple, 1 for orange
labels = []
# Features of the fruit: weight and texture
features =  []

example = []
pil_image1 = Imag
e.open("./images/Orange1.jpg")
opencvImage1 = cv.cvtColor(numpy.array(pil_image1), cv.COLOR_BGR2GRAY)
gray_image1 = opencvImage1.reshape(-1, opencvImage1.shape[-1])  
gray_image1 = list(itertools.chain(*gray_image1)) 
example.append(gray_image1)
#print(example)
for item in files:
    if item[0] == 'A':
        labels.append("Apple")
    else:
        labels.append("Orange")
    
    pil_image = Image.open(f"./images/{item}")
    opencvImage = cv.cvtColor(numpy.array(pil_image), cv.COLOR_BGR2GRAY)
    gray_image = opencvImage.reshape(1,-1)
    gray_image = list(itertools.chain(*gray_image)) 
    features.append(gray_image)


# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier on the features and labels

clf = clf.fit(features, labels)
# Predict the class of a new fruit with weight 160g and texture 0 (orange)
print(clf.predict(example))