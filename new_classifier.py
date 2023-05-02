from sklearn import tree
from PIL import Image
import cv2 as cv
import numpy
import os 
import itertools
# <<---  --->>      ==>     Superfluous calculation
files = os.listdir('./images')

#print(files)
# Labels for the features: 0 for apple, 1 for orange
labels = []
# Features of the fruit: weight and texture
features =  []
# An example of how a neural network works
example = []
# Example image for neural network 
pil_image1 = Image.open("./images/Orange2.jpg")
# Convert to number-array 
opencvImage1 = cv.cvtColor(numpy.array(pil_image1), cv.COLOR_BGR2GRAY)
#gray_image1 = opencvImage1.reshape(-1, opencvImage1.shape[-1])  
# Unpacking array
gray_image1 = list(itertools.chain(*opencvImage1)) 
# Append result into example variable
example.append(gray_image1)
#print(example)
for item in files:
    # <<---  Adding responses for a neural network   --->>
    if item[0] == 'A':
        labels.append("Apple")
    else:
        labels.append("Orange")
    # An array of training examples 
    pil_image = Image.open(f"./images/{item}")
    opencvImage = cv.cvtColor(numpy.array(pil_image), cv.COLOR_BGR2GRAY)
    #<<---   gray_image = opencvImage.reshape(1,-1)   --->> 
    # Unpacking array
    gray_image = list(itertools.chain(*opencvImage)) 
    features.append(gray_image)


# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier on the features and labels
clf = clf.fit(features, labels)
# Predict the class of a new fruit with weight 160g and texture 0 (orange)
print(clf.predict(example))