#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import math
import statistics
import numpy as np
import scipy.stats
import seaborn as sns  
from matplotlib import pyplot as plt 
from matplotlib import colors as col
from matplotlib.ticker import PercentFormatter as pcrt


# In[2]:


# read data 
data = pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv",low_memory=False) 
data.head(3)


# In[413]:


data.shape


# In[3]:


############################################## EDA
# check data shape
print(data.shape)
# check data types 
data.dtypes


# In[5]:


# check and remove the duplicate rows
duplicate_rows_df = data[data.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
df = data.drop_duplicates()
# check the count numbers for all cols


# In[6]:


# check the count numbers for all cols
df.count()


# In[7]:


# check missing values
print(df.isnull().sum())


# In[8]:


# delete the cols have tons of missings also not relative for modeling in heathcare
df = df.drop(['Payment Typology 2', 'Payment Typology 3', 'Zip Code - 3 digits', 'Operating Certificate Number', 'Facility Id'], axis=1)


# In[9]:


# Dropping the missing values.
df = df.dropna()    
df.count()


# In[10]:


# check the statistic summary of data
df.describe()


# In[11]:


# Creating histogram 
fig, axs = plt.subplots(1, 1, 
                        figsize =(10, 7),  
                        tight_layout = True) 
  
axs.hist(df["Total Costs"], bins = 20) 
  
# Show plot We found the response total costs is super right skewed.
plt.show() 


# In[12]:


# Check 0 value for total costs
zero=df["Total Costs"]==0
zero.value_counts()


# In[13]:


# Remove 0 values in total costs
df_new=df[-zero]
df_new.shape


# In[14]:


# Creating histogram on log total cost 
fig, axs = plt.subplots(1, 1, 
                        figsize =(10, 7),  
                        tight_layout = True) 
  
axs.hist(np.log(df_new["Total Costs"]), bins = 20) 
  
# Show plot We found the response total costs is super right skewed.
plt.show() 


# In[15]:


################################################ Feature Associations
# check some categorical variables' relationship with total costs
df_new.groupby(["Gender"]).mean()


# In[16]:


df_new.groupby(["Race"]).mean()


# In[17]:


df_new.groupby(["Type of Admission"]).mean()


# In[18]:


df_new.groupby(["Emergency Department Indicator"]).mean()


# In[19]:


df_new.groupby(["Emergency Department Indicator"]).mean()


# In[20]:


df_new.groupby(["Payment Typology 1"]).mean()


# In[21]:


plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# In[22]:


df_new.plot.scatter(x="Birth Weight",y="Total Costs")


# In[23]:


df_new.dtypes


# In[180]:


# Remove and Clean variables in the model
df_new2 = df_new.drop(['Discharge Year','Facility Name','CCS Diagnosis Code','CCS Procedure Code', 
                       'APR DRG Code', 'APR MDC Code','APR Severity of Illness Code',
                       'Abortion Edit Indicator','Total Charges','APR DRG Description',
                       'Hospital County',#"CCS Procedure Description",'CCS Diagnosis Description',
                       'APR MDC Description'], axis=1)


# In[181]:


# Remove the + string in Length of Stay
df_new2["Length of Stay"] = df_new2["Length of Stay"].map(lambda x: x.rstrip(' +'))


# In[182]:


# change object to float for Length of Stay
df_new2["Length of Stay"]=pd.to_numeric(df_new2["Length of Stay"])


# In[246]:


diag_desc_table = df_new2.groupby('Gender')['Gender'].count()
diag_desc_table.sort_values()


# In[247]:


df_new2 = df_new2[df_new2['Gender'] != "U"]


# In[248]:


diag_desc_table = df_new2.groupby('CCS Diagnosis Description')['CCS Diagnosis Description'].count()
diag_desc_table.sort_values()


# In[213]:


# remove levels with a very small number of records (<50 for 10%)
df_new2 = df_new2[df_new2['CCS Diagnosis Description'] != "Female infertility"]
df_new2 = df_new2[df_new2['CCS Diagnosis Description'] != "Contraceptive and procreative management"]
df_new2 = df_new2[df_new2['CCS Diagnosis Description'] != "Osteoporosis"]
df_new2 = df_new2[df_new2['CCS Diagnosis Description'] != "Cataract"]


# In[223]:


diag_desc_table = df_new2.groupby('CCS Procedure Description')['CCS Procedure Description'].count()
diag_desc_table.sort_values()


# In[222]:


# remove levels with a very small number of records (<50 for 10%)
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "INJ/LIG ESOPH VARICES"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "CORONARY THROMBOLYSIS"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "MAMMOGRAPHY"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "TYMPANOPLASTY"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "CORNEAL TRANSPLANT"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "INTRAOP CHOLANGIOGRAM"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "NONOP URINARY SYS MEASR"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "VARI VEIN STRIP;LOW LMB"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "GLAUCOMA PROCEDURES"]
df_new2 = df_new2[df_new2['CCS Procedure Description'] != "DX AMNIOCENTESIS"]


# In[224]:


# Metric Selection 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso


# In[281]:


print(df_new2.shape)
df_new2.dtypes


# In[289]:


# Split the X and Y 
X = df_new2.iloc[:, 0:16]
y = df_new2.iloc[:, 16]


# In[290]:


print(X.shape)
print(y.shape)


# In[300]:


seed = 50  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.90, random_state = seed)


# In[301]:


# One-Hot encode
features_to_encode = X_train.columns[X_train.dtypes==object].tolist()  
features_to_encode 


# In[302]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
col_trans = make_column_transformer(
                        (OneHotEncoder(),features_to_encode),
                        remainder = "passthrough"
                        )
col_trans


# In[303]:


# Random Forest model
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(
                     min_samples_leaf=50,
                     n_estimators=150,
                     bootstrap=True,
                     oob_score=True,
                     n_jobs=-1,
                     random_state=seed,
                     max_features='auto')


# In[304]:


from sklearn.pipeline import make_pipeline
import datetime 
start=datetime.datetime.now()

pipe = make_pipeline(col_trans, rf_reg)
pipe.fit(X_train, y_train)
         
end=datetime.datetime.now()
time_diff=end-start
print( "Time=" + str(time_diff.seconds))


# In[305]:


# Model performance Test error 
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[306]:


#predictions = pipe.predict(X_test)
#predictions
#y_test


# In[384]:


test_error = evaluate(pipe, X_test, y_test)


# In[385]:


# baseline compare
train_y_mean=np.mean(y_train)
np.mean(abs(train_y_mean-y_test))


# In[386]:


# get importance
importance = pipe.steps[1][1].feature_importances_
names=pipe.steps[0][1].get_feature_names()


# In[387]:


# create a dictionary
ex_dic = {
    'names': names,
    'importance_value': importance
}

# create a list of strings
columns = ['names', 'importance_value']

#index = ['a', 'b', 'c']

# Passing a dictionary
# key: column name
# value: series of values
variable_importance_df = pd.DataFrame(ex_dic, columns=columns)
#variable_importance_df.names
variable_importance_df.names = variable_importance_df.names.str.replace('__','_')
variable_importance_df.names = variable_importance_df.names.str.replace('onehotencoder','')
variable_importance_df.names = variable_importance_df.names.str.lstrip('_')
variable_importance_df[['vars','level']] = variable_importance_df.names.str.split("_",expand=True) 

#variable_importance_df.groupby('vars')['vars'].count()


# In[388]:


names_list=list(df_new2.columns)
names_list.remove("Length of Stay")
names_list.remove("Birth Weight")
names_list.remove("Total Costs")
#print(names_list)


# In[389]:


variable_importance_df.vars = variable_importance_df.vars.str.replace('x10',names_list[10])
variable_importance_df.vars = variable_importance_df.vars.str.replace('x11',names_list[11])
variable_importance_df.vars = variable_importance_df.vars.str.replace('x12',names_list[12])
variable_importance_df.vars = variable_importance_df.vars.str.replace('x13',names_list[13])
for i in range(10):
    #print('x' + str(i),names_list[i])
    variable_importance_df.vars = variable_importance_df.vars.str.replace('x' + str(i),names_list[i])

#variable_importance_df.vars.head(3)
#variable_importance_df[variable_importance_df['vars'] == "x10"]

#variable_importance_df.groupby('vars')['vars'].count()

#variable_importance_df.vars = variable_importance_df.vars.str.replace("Age Group0","Age Group")
#variable_importance_df.vars = variable_importance_df.vars.str.replace("Age Group1","Age Group")
#variable_importance_df.vars = variable_importance_df.vars.str.replace("Age Group2","Age Group")
#variable_importance_df.vars = variable_importance_df.vars.str.replace("Age Group3","Age Group")
#variable_importance_df.vars


# In[390]:


feature_importance = variable_importance_df.groupby(["vars"]).sum()
feature_importance2 = feature_importance.sort_values(by=["importance_value"],ascending=False)
print(feature_importance2)


# In[391]:


# check levels
# variable_importance_df[variable_importance_df['vars'] == "CCS Procedure Description"]


# In[405]:


from sklearn.inspection import plot_partial_dependence
fig, ax = plt.subplots(figsize = (12, 6))
# model name is pipe
#X_train.columns
plot_partial_dependence(pipe, X_train, [5], grid_resolution = 2000, n_jobs = -1, ax=ax) 


# In[ ]:





# In[ ]:





# In[ ]:


#####################################################################################
#################################### cross validation ###############################
#####################################################################################


# In[407]:


# Model hyperparameter optimization - k fold cross validation
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint


# In[408]:


# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[409]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 5 )]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
pprint(random_grid)


# In[410]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
# rf_random.fit(train_features, train_labels)


# In[411]:


print(X_train.shape)
print(y_train.shape)


# In[412]:


pipe = make_pipeline(col_trans, rf_random)
pipe.fit(X_train, y_train)


# In[429]:


best_random = pipe.steps[1][1].best_estimator_
best_random


# In[434]:


rf_reg2 = RandomForestRegressor(
                     min_samples_leaf=2,
                     n_estimators=650,
                     bootstrap=True,
                     oob_score=True,
                     n_jobs=-1,
                     random_state=seed,
                     max_depth=32,
                     max_features='auto')


# In[ ]:


start=datetime.datetime.now()
pipe = make_pipeline(col_trans, rf_reg2)
pipe.fit(X_train, y_train)
         
end=datetime.datetime.now()
time_diff=end-start
print( "Time=" + str(time_diff.seconds))


# In[433]:


test_error = evaluate(pipe, X_test, y_test)


# In[432]:


pipe.steps[1][1].cv_results_


# In[ ]:


random_accuracy = evaluate(pipe, X_test, y_test)


# In[32]:


# Timing for one variable models
import datetime 
for c in range(14):
    start=datetime.datetime.now()
    X = pd.DataFrame(df_new2.iloc[:, c])
    y = df_new2.iloc[:, 14]
    seed = 50  
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state = seed)
    # One-Hot encode
    features_to_encode = X_train.columns[X_train.dtypes==object].tolist()  
    #features_to_encode 
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer
    col_trans = make_column_transformer(
                            (OneHotEncoder(),features_to_encode),
                            remainder = "passthrough"
                            )
    #col_trans
    # Random Forest model
    from sklearn.ensemble import RandomForestRegressor
    rf_reg = RandomForestRegressor(
                         min_samples_leaf=50,
                         n_estimators=150,
                         bootstrap=True,
                         oob_score=True,
                         n_jobs=-1,
                         random_state=seed,
                         max_features='auto')
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(col_trans, rf_reg)
    pipe.fit(X_train, y_train)
    end=datetime.datetime.now()
    time_diff=end-start
    print("Col="+ str(c) + " Time=" + str(time_diff.seconds))

