import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
"""from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO """


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
#print(dataset_UP.columns)
X=dataset_UP.iloc[:,:-2]
Y=dataset_UP.iloc[:,8]#print(X.tail(4))
features=dataset_UP.columns[:-1]

"""for i in range(7):
    plt.hist(X[:,i],edgecolor="black")
    plt.title(features[i])
    plt.show()"""


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=SVC()
model.fit(X_train,Y_train)
predictions=model.predict(X_test)
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print("Accuracy on training dataset",model.score(X_train,Y_train))
print("Accuracy on test dataset",model.score(X_test,Y_test ))


# Parameter Tuning
kernel=["linear","rbf","poly","sigmoid"]
for i in kernel:
    model=SVC(kernel=i,C=1.0)
    model.fit(X_train,Y_train)
    print("For kernel",i)
    print("Accuracy is",model.score(X_test,Y_test))
param_dist={'C':[0.1,1,100,1000],'kernel':["rbf","poly","linear","sigmoid"],"degree":[1,2,3,4,5]}
model=SVC()
svm_cv=RandomizedSearchCV(model,param_dist,cv=3)
svm_cv.fit(X_train,Y_train)
print("Tuned Decision Tree parameters:{}".format(svm_cv.best_params_))
print("Best score is:{}".format(svm_cv.best_score_))

