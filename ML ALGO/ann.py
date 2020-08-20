import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils



mydataset=pd.read_csv("city_hour.csv")
mydataset.interpolate(limit_direction="both",inplace=True)
#mydataset['BTX']=mydataset['Benzene']+mydataset['Toluene']+mydataset['Xylene']
#mydataset['Nitrogen_compounds']=mydataset['NO']+mydataset['NO2']+mydataset['NOx']+mydataset['NH3']
#print(mydataset.head(4))
#mydataset.drop(['Benzene','Toluene','Xylene','Datetime'],axis=1,inplace=True)
#mydataset.drop(['NO','NOx','NO2','NH3'],axis=1,inplace=True)
#print(mydataset.head(1))
mydataset.set_index('City',inplace=True)
cities_undertaken=['Delhi']
df=mydataset[['AQI','AQI_Bucket']]
mydataset.drop(['AQI','AQI_Bucket',"Datetime"],axis=1,inplace=True)
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
print(dataset_UP.columns)
X=dataset_UP.iloc[:,:-2]
Y=dataset_UP.iloc[:,13]
print(Y.tail(4))
features=dataset_UP.columns[:-1]
# Encode the the output variables ,one hot encoding
encoder=LabelEncoder()
encoder.fit_transform(Y)

Y=pd.get_dummies(Y).values


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
#Splitting into validation dataset
#X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.25,random_state=10)
#print(dataset_UP.shape,X_train.shape,X_val.shape,X_test.shape)"""
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# -*- coding: utf-8 -*-"""
"""sns.countplot(x="AQI",data=dataset_UP)
dataset_UP.corr()["AQI"].sort_values().plot(kind="bar")
sns.heatmap(dataset_UP.corr())"""



model=Sequential()
model.add(Dense(12,input_shape=(12,),activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(6,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x=X_train,y=Y_train,epochs=250,verbose=0)
loss_df=pd.DataFrame(model.history.history)
loss_df.plot()


#print(model.evaluate(X_train,Y_train))
"""test_predictions=model.predict(X_test)
test_predictions=pd.Series(test_predictions.reshape(9346,))
pred_df=pd.DataFrame(Y_test,columns=["TestTrue"])
pred_df.concat([pred_df,test_predictions],axis=1)
pred_df.columns=["TestTrue","Model Predictions"]
sns.scaterplot(x="Test True",y="Model Predictions",data=pred_df)
"""

"""y_test_class=np.argmax(Y_test,axis=1)
y_pred_class=np.argmax(test_predictions,axis=1)"""
#print(test_predictions)

"""print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))"""

