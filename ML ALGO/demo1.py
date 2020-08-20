import pandas as pd
def missing_values(df):
    mis_val=df.isnull().sum()
    mis_val_percent=100*mis_val/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n""There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
    print(mis_val_table_ren_columns)
        
    
mydataset=pd.read_csv("city_hour.csv",parse_dates=['Datetime'])
#print(mydataset)
print(mydataset.head())
mydataset.interpolate(limit_direction="both",inplace=True)
print("After missing Values")
print(mydataset.tail())
#splitting the data set into dependent and independent
#handling missing values
missing_values(mydataset)
X=mydataset.iloc[:,:-2]
Y=mydataset.iloc[:,15:17]
print(X.tail())
print(Y.head())
cities = mydataset['City'].value_counts()
print(cities.index)
#mydataset['Datetime'] = pd.to_datetime(mydataset['Datetime'])
#print(mydataset.head())
#combining BTX
mydataset['BTX']=mydataset['Benzene']+mydataset['Toluene']+mydataset['Xylene']
print(mydataset.head())
mydataset.drop(['Benzene','Toluene','Xylene'],axis=1,inplace=True)
print(mydataset.head(1))
print(type(mydataset['Datetime']))