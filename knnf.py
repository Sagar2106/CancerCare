import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib is used for plot the graphs,
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/output_knn_train.csv")
data["Diagnosis"]=data["File Name"].map({'Bengincase':0,'Malignantcase':1,'Normalcase':2}).astype(int)
data=data.drop(["File Name"],axis=1)
data.to_csv(r'C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/output_knn_working_train.csv', index = False)

data=pd.read_csv("C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/output_knn_test.csv")
data["Diagnosis"]=data["File Name"].map({'Bengincase':0,'Malignantcase':1,'Normalcase':2}).astype(int)
data=data.drop(["File Name"],axis=1)
data.to_csv(r'C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/output_knn_working_test.csv', index = False)


data1=pd.read_csv("C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/output_knn_working_train.csv")
data2=pd.read_csv("C:/Users/Vihaan/Desktop/Project/Filtered GLCM and kNN/output_knn_working_test.csv")

print(data1.shape)
print(data2.shape)

x1 = data1[['Energy', 'Corr', 'Diss_sim', 'Homogen', 'ASM', 'Energy2', 'Corr2',
       'Diss_sim2', 'Homogen2', 'ASM2', 'Energy3', 'Corr3', 'Diss_sim3',
       'Homogen3', 'ASM3', 'Energy4', 'Corr4', 'Diss_sim4', 'Homogen4', 'ASM4',
       'Entropy']]
y1 = data1[['Diagnosis']]

x2 = data2[['Energy', 'Corr', 'Diss_sim', 'Homogen', 'ASM', 'Energy2', 'Corr2',
       'Diss_sim2', 'Homogen2', 'ASM2', 'Energy3', 'Corr3', 'Diss_sim3',
       'Homogen3', 'ASM3', 'Energy4', 'Corr4', 'Diss_sim4', 'Homogen4', 'ASM4',
       'Entropy']]
y2 = data2[['Diagnosis']]

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
x_train, y_train = x1, y1
x_test, y_test = x2, y2


model = KNeighborsClassifier(n_neighbors = 8)
model.fit(x_train, y_train)
predict = model.predict(x_test)

accuracy=model.score(x_train,y_train)
print(accuracy)
accuracy2=model.score(x_test,y_test)
print(accuracy2)

'''
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=2)
print(score)
'''
print(predict)