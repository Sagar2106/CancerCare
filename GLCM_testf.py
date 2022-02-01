import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from csv import writer
from csv import reader
import re

#Resize images to
SIZE = 512

#Capture images and labels into arrays.
#Start by creating empty lists.
train_images = []
train_labels = [] 
label_name = []
#for directory_path in glob.glob("cell_images/train/*"):
for directory_path in glob.glob("C:/Users/Vihaan/Desktop/Project/Test Cases Filtered/"):
    label = directory_path.split("\\")[-1] 
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
       # print("Hello!")
        temp = os.path.basename(img_path)
        filename, file_extension = os.path.splitext(os.path.basename(img_path))
        print(filename)
        new_name3 = re.sub('[0-9]', '', filename)
        new_name2 = re.sub('\(', '', new_name3)
        new_name1 = re.sub('\)', '', new_name2)
        new_name = re.sub(' ', '', new_name1)
        print(new_name)
        #label_name.append(os.path.basename(img_path))
        label_name.append(new_name)
        print(img_path)
        
        img = cv2.imread(img_path, 0) #Reading color images
        img = cv2.resize(img, (SIZE, SIZE)) #Resize images
        train_images.append(img)
        #train_labels.append(label)
        train_labels.append(label)
    #print(train_images)


train_images = np.array(train_images)
train_labels = np.array(train_labels)

#Do exactly the same for test/validation images
# test
test_images = []
test_labels = []
#for directory_path in glob.glob("cell_images/test/*"): 
for directory_path in glob.glob("C:/Users/Vihaan/Desktop/Project/Test Cases Filtered/"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (SIZE, SIZE))
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
#If you only have one dataset then split here
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
#x_train, x_test = x_train / 255.0, x_test / 255.0

###################################################################
# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        #print(image)
        #print("Hello!")
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        img = dataset[image, :,:]
        
    ################################################################
    #START ADDING DATA TO THE DATAFRAME
  
                
         #Full image
        #GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM = greycomatrix(img, [1], [0])       
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss       
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_asm = greycoprops(GLCM, 'ASM')[0]
        df['ASM'] = GLCM_asm
       # GLCM_contr = greycoprops(GLCM, 'contrast')[0]
       # df['Contrast'] = GLCM_contr


        GLCM2 = greycomatrix(img, [1], [np.pi/2])       
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2      
        GLCM_asm2 = greycoprops(GLCM, 'ASM')[0]
        df['ASM2'] = GLCM_asm2
        #GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        #df['Contrast2'] = GLCM_contr2

        GLCM3 = greycomatrix(img, [1], [np.pi/4])       
        GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3       
        GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3       
        GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3   
        GLCM_asm3 = greycoprops(GLCM, 'ASM')[0]
        df['ASM3'] = GLCM_asm3
        #GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        #df['Contrast3'] = GLCM_contr3
        
        #GLCM4 = greycomatrix(img, [0], [np.pi/4])  
        GLCM4 = greycomatrix(img, [1], [3*np.pi/4])       
        GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4       
        GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4       
        GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4 
        GLCM_asm4 = greycoprops(GLCM, 'ASM')[0]
        df['ASM4'] = GLCM_asm4
        #GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        #df['Contrast4'] = GLCM_contr4
        '''
        #GLCM5 = greycomatrix(img, [0], [np.pi/2])      
        GLCM5 = greycomatrix(img, [9], [np.pi/4])  
        GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5       
        GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5       
        GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5       
        GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5
        '''
        #Add more filters as needed
        entropy = shannon_entropy(img)
        df['Entropy'] = entropy

        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        
    return image_dataset
####################################################################
#Extract features from training images
image_features = feature_extractor(x_train)
X_for_ML = image_features

#saving file to csv
label_dataframe = pd.DataFrame(label_name,columns=['File Name'])
label_dataframe.to_csv(r'C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/file_name_noseq.csv', index = False, header = True)
image_features.to_csv(r'C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/test_train.csv', index = False, header = True)
df = pd.read_csv("C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/test_train.csv")
df["File Name"] = pd.read_csv('C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/file_name_noseq.csv')
df.to_csv(r'C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/output_knn_test.csv', index = False)
