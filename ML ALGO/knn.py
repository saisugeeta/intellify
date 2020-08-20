# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline
mydataset=pd.read_csv("city_hour.csv")
mydataset.interpolate(limit_direction="both",inplace=True)
mydataset['BTX']=mydataset['Benzene']+mydataset['Toluene']+mydataset['Xylene']
mydataset['Nitrogen_compounds']=mydataset['NO']+mydataset['NO2']+mydataset['NOx']+mydataset['NH3']
#print(mydataset.head(4))

mydataset.drop(['Benzene','Toluene','Xylene','Datetime'],axis=1,inplace=True)
mydataset.drop(['NO','NOx','NO2','NH3'],axis=1,inplace=True)
#print(mydataset.head(1))
mydataset.set_index('City',inplace=True)
cities_undertaken=['Delhi']
df=mydataset[['AQI','AQI_Bucket']]
mydataset.drop(['AQI','AQI_Bucket'],axis=1,inplace=True)
mydataset['AQI']=df['AQI']

#mydataset['AQI_Bucket']=df['AQI_Bucket']
dataset_UP=mydataset.loc[cities_undertaken]
# Adding extra col to divide as classes
column_name='AQI'
conditions=[dataset_UP[column_name]>400,(dataset_UP[column_name]<400) & (dataset_UP[column_name]>300),
            (dataset_UP[column_name]<300) &(dataset_UP[column_name]>200),
            (dataset_UP[column_name]<200)&(dataset_UP[column_name]>100),
            (dataset_UP[column_name]<100)&(dataset_UP[column_name]>50),(dataset_UP[column_name]<50)]
classes=["severe","verypoor","poor","moderate","satisfactory","good"]
dataset_UP["classes_col"]=np.select(conditions,classes,default="moderate")
#sns.pairplot(dataset_UP,hue="AQI")
print(dataset_UP.columns)
X=dataset_UP.iloc[:,:-2]
Y=dataset_UP.iloc[:,8]
scaler=StandardScaler()
scaler.fit(X)
scaled_features=scaler.transform(X)
scaled_X=pd.DataFrame(scaled_features,columns=X.columns)
print(Y.head())



# Splitting into training and test dataset
X_train,X_test,Y_train,Y_test=train_test_split(scaled_X,Y,test_size=0.2,random_state=10)
#Splitting into validation dataset
#X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.25,random_state=10)
#print(dataset_UP.shape,X_train.shape,X_val.shape,X_test.shape)
#knn algo
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)
prediction=knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,prediction))
print(classification_report(Y_test,prediction))
error_rate=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    prediction_i=knn.predict(X_test)
    error_rate.append(np.mean(prediction_i!=Y_test))
plt.figure(figsize=(10,10))    
plt.plot(range(1,21),error_rate,color="blue",linestyle="dashed",marker="o",markerfacecolor="red",markersize=10)
plt.title("Errorrate Vs K")
plt.xlabel("K")
plt.ylabel("Error_rate")
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
prediction=knn.predict(X_test)
print(confusion_matrix(Y_test,prediction))
print(classification_report(Y_test,prediction))
distance_metrics="euclideandistance"


     














