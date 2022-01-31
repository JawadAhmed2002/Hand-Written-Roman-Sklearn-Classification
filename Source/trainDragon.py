import os
import sys
from matplotlib import image as maImg
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.thresholding import threshold_mean
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.feature import canny
from skimage.filters import threshold_otsu
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import skimage.io
import skimage.transform


np.set_printoptions(threshold=sys.maxsize)
# Read image files of given directory


def readImages(dirPath):
    all_images = list()
    imagesCount = []
    for (path, names, filenames) in sorted(os.walk(dirPath)):
        imagesCount.append(len(filenames))
        all_images += [os.path.join(path, file) for file in filenames]
    imagesCount = [i for i in imagesCount if i != 0]
    return all_images, imagesCount

# Extract Features From the image arrays

def featureExtraction(Images, feature):
    featureExtract = []
    for i in Images:
        imageRead = maImg.imread(i)
        # Count only darker part of an image
        if feature == 'blackPart':
            imageRead = np.round(imageRead)
            count = np.count_nonzero(imageRead == 0)
            featureExtract.append(count)
        # extracting the edge
        elif feature == 'edgeDetection':
            count = 0
            img = canny(imageRead)
            count = np.count_nonzero(imageRead == 0)
            featureExtract.append(count)
        #Extracting using threshold
        elif feature == 'Imgthreshold':
            threshol_val = threshold_mean(imageRead)
            threshold_img = imageRead>threshol_val
            data = np.count_nonzero(threshold_img == False)
            img_size = imageRead.shape
            count = (img_size[0]*img_size[1])-data
            featureExtract.append(data)

        elif feature == 'diagonal':
            imageRead = np.resize(imageRead, (300, 300))
            left_diag=[imageRead[j][j] for j in range(len(imageRead))]  
            right_diag=[imageRead[len(imageRead)-1-i][i] for i in range(len(imageRead)-1,-1,-1)]
            # left_diag=np.mean(left_diag)
            # right_diag=np.mean(right_diag)
            mid_pixel = len(left_diag)//2
            new_img = small[mid_pixel-10:mid_pixel+10]
            featureExtract.append([new_img,right_diag])
            
            # featureExtract.append([left_diag,right_diag])
        # Transforming the images into smaller image
        elif feature=='transformation':
            # read in image
            image = np.resize(imageRead, (300, 300))
            # resize the image
            new_shape = (image.shape[0] // 2, image.shape[1] // 2)
            small = skimage.transform.resize(image=image, output_shape=new_shape)
            center = len(small)//2
            img = small[center-5:center+50, center+5:center+50]
            featureExtract.append(img)
        
        elif feature=='centroid':
            mid_pixel = len(imageRead)//2
            new_img = imageRead[mid_pixel-5:mid_pixel+5]
            train_img = []
            for i in new_img:
                train_img.append(i[(len(i)//2)-50:(len(i)//2)+50])
            train_img = np.array(train_img)
            train_img = np.reshape(train_img, (1, 1000))
            # print(train_img)
            featureExtract.append(train_img)
    return featureExtract


# get all train data
trainData, trainImagesCount = readImages("data/train/")
# get all val data
valData, valImagesCount = readImages("data/val/")
#__________________________________________________________________________________________________________________#

# Xtrain of darker part images
# xTrain = featureExtraction(trainData,'blackPart')

# Xtrain of edge part of images
# xTrain = featureExtraction(trainData,'edgeDetection')

# Xtrain of applying threshold to an images
# xTrain = featureExtraction(trainData,'Imgthreshold')

# Xtrain of diagonal part of images
# xTrain = featureExtraction(trainData, 'diagonal')

# Xtrain of diagonal part of images
# xTrain = featureExtraction(trainData, 'diagonal')

# Xtrain of tranforming image into smallar parts
# xTrain = featureExtraction(trainData, 'transformation')

# Xtrain of centroid of images
xTrain = featureExtraction(trainData,'centroid')
# print(xTrain)


xTrain = np.array(xTrain)
xTrain = xTrain.reshape(len(xTrain), -1)
# Y train Desired output, it will map the train data
yTrain = []
digits = 1
for i in trainImagesCount:
    for j in range(i):
        yTrain.append(digits)
    digits += 1
# print(yTrain)
yTrain = np.array(yTrain)
yTrain = yTrain.reshape(len(yTrain), -1)
yTrain = yTrain.ravel()

# xTest = xTrain
# yTest = yTrain


# Xtest of central part ofimagesCount images
xTest = featureExtraction(valData,'centroid')
# print(xTrain)
xTest = np.array(xTest)
xTest = xTest.reshape(len(xTest), -1)
# Y test Desired output, it will map the train data
yTest = []
digits = 1
for i in valImagesCount:
    for j in range(i):
        yTest.append(digits)
    digits += 1
# print(yTest)
yTest = np.array(yTest)
yTest = yTest.reshape(len(yTest), -1)
#________________________________________________________________________________________________________________________________________________________________________________________________________#

# Use Multi Layer Perceptron Classifier
clf = MLPClassifier(hidden_layer_sizes=(100, 50, 10), activation='logistic', solver="sgd", learning_rate_init=0.1,
                    random_state=1, verbose=True, max_iter=800, n_iter_no_change=200).fit(xTrain, yTrain)

result = clf.predict(xTest)
result = np.array(result)
print(result)

# Check out the accuracy of Train X with Train Y mean Training Data Recognizing Accuracy After MLP Training
print('Accuracy Of train_X with train_Y is:',(round((clf.score(xTrain, yTrain)),2)*100),"%")
# Check out the accuracy of Test X with Test Y mean Validation Data Recognizing Accuracy After MLP Training
print('Accuracy Of test_X with test_Y is:',(round((clf.score(xTest, yTest)),2)*100),'%')






