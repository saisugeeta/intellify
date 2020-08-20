# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:06:59 2020

@author: Sugeeta
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
"""from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO """
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

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
print(dataset_UP.columns)
X=dataset_UP.iloc[:,:-2]
Y=dataset_UP.iloc[:,8]#print(X.tail(4))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#print(Y.head(4))
# Splitting into training and test dataset

#Splitting into validation dataset
#X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.25,random_state=10)
#print(dataset_UP.shape,X_train.shape,X_val.shape,X_test.shape)
dtree=DecisionTreeClassifier(random_state=0)
dtree=dtree.fit(X_train,Y_train)
predictions=dtree.predict(X_test)

print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))

#Parameter tuning
param_dist={"max_depth":[3,None],"min_samples_leaf":randint(1,9),"criterion":["gini","entropy"]}
tree_cv=RandomizedSearchCV(dtree,param_dist,cv=5)
tree_cv.fit(X_train,Y_train)
print("Tuned Decision Tree parameters:{}".format(tree_cv.best_params_))
print("Best score is:{}".format(tree_cv.best_score_))

# RandomForest

rfc=RandomForestClassifier()
rfc.fit(X_train,Y_train)
rfcpred_proba=rfc.predict_proba(X_test)
#print(rfcpred_proba)
rfc_pred=rfc.predict(X_test)
print(classification_report(Y_test,rfc_pred))
print(confusion_matrix(Y_test,rfc_pred))

#Parameter tuning
#n_estimators=[int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
param_dist={"max_depth":[3,None],"min_samples_leaf":randint(1,9),"criterion":["gini","entropy"],"n_estimators":[100,200,300,400,500]}
tree_cv=RandomizedSearchCV(rfc,param_dist,cv=3)
tree_cv.fit(X_train,Y_train)
print("Tuned Decision Tree parameters:{}".format(tree_cv.best_params_))
print("Best score is:{}".format(tree_cv.best_score_))
best_random = tree_cv.best_estimator_
random_accuracy = evaluate(best_random, X_test, Y_test)


plt.figure(figsize=(25,10))
#tree.plot_tree(dtree,filled=True)
#plt.show()


