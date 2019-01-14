'''
This script performs feature selection, applying three consecutive methods:
    - Removal of features with low variance - features that were either 1 or 0 in at least 80\% of the observations, were excluded;
    - Removal of highly correlated features - in pairs of features with Pearson a correlation coefficient higher than 98%, one of the features is removed
    - Seletion of the 60 features with the highest importance, according to the Extremely Randomized Trees Classifier
'''
# %%
#Importing libraries
import pickle
from sklearn.feature_selection import VarianceThreshold
import pywt
import os
import numpy as np
import scipy.stats as spstat
import matplotlib.pyplot as plt
from collections import OrderedDict
from astropy.table import Table
import csv
import pandas as pd 
import matplotlib.pyplot as plot
import sys
import os
from collections import Counter, OrderedDict
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from astropy.table import Table
from cesium.time_series import TimeSeries
import cesium.featurize as featurize
from tqdm import tnrange, tqdm_notebook
import sklearn 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ConstantKernel, WhiteKernel)
from feature_selector import FeatureSelector 
import warnings
warnings.simplefilter('ignore')
sys.path.append(r'C:\Users\maria\Desktop\plasticc-kit-master')
# %%
'''Selection of the final feature selection strategie (PCA (discarded) vs Feature Importance)'''
#FS_final_strategie='pca'
FS_final_strategie='feature_importance'
# %%
'''Opening the pickle file containing the features, without any processing'''
with open(r'C:\Users\maria\Desktop\plasticc-kit-master\wavelets_features_all_decomps.pickle', 'rb') as handle:
    feature_matrix=pickle.load(handle)


'''Removing the features which have either NAN or Inf values'''
feature_matrix_without_nans=OrderedDict()
print('Removing invalid features (nans/infs)...')

#Converting the Inf values to NANs and, for each observation, removing the features which contain NANs
for i in range(0,7848):
    df=feature_matrix[i]
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna(axis='columns')
    feature_matrix_without_nans[i]=df

#The same feature can have value NAN in an observation and have a valid value in another, so, after the NAN removal, we only keep the features which are common to all observations.
#In the cycle below, we get the list of common features
df=feature_matrix_without_nans[0]    
col_list=df.columns    
for i in range(1,7848):    
    df=feature_matrix_without_nans[i]  
    col_list=set(col_list) & set(df.columns)
      
#Getting feature matrix with the common features
feature_matrix=np.zeros([7848,len(col_list)])
col_list=list(col_list)
for i in range(0,7848): 
    df=feature_matrix_without_nans[i]  
    feature_matrix[i,:]=df[col_list].values
    
print('Finished!')    
 # %% 
'''Opening the metadata''' 
from sklearn.ensemble import ExtraTreesClassifier
meta_data = pd.read_csv('C:/Users/maria/Desktop/plasticc-kit-master/data/training_set_metadata.csv')
target_vec = meta_data['target'] #vector with the labels of each observation
# %% 
'''Merging the features with the metadata features which do not contain NAN or Inf values'''
features = pd.DataFrame(data=feature_matrix)
features=pd.concat([features, meta_data.iloc[:,[1,2,3,4,5,6,7,8,10]]], axis=1) 

'''Removing the features with low variance (features that are 1 or 0 for at least 80% of the observations)''' 
sel = VarianceThreshold(threshold=(.8 * (1 - .8))) #Boolean features are Bernoulli random variables, and the variance of such variables is given by p(1-p)
features=sel.fit_transform(features)
# %%
'''Removing highly correlated features'''
features = pd.DataFrame(data=features)
fs = FeatureSelector(data = features, labels = target_vec)
fs.identify_collinear(correlation_threshold = 0.98)
fs.plot_collinear()

# list of collinear features to remove
collinear_features = fs.ops['collinear']

# dataframe of collinear features
fs.record_collinear.head()

# Set of features after the removal of the highly correlated features
train_removed = fs.remove(methods = 'all')
# %% 
if FS_final_strategie=='feature_importance':
    #--------------------------OPTION 1: FEATURE IMPORTANCE------------------------
    #Fitting the Extremely Randomized Trees Classifier
    model = ExtraTreesClassifier()
    model.fit(train_removed, target_vec)
    
    #getting the feature importances
    importances=model.feature_importances_
    #print(model.feature_importances_)
    
    #Sorting the features according to their importances
    indices = np.argsort(importances)[::-1]
    #print(len(indices))
    
    #Selecting the 60 most important features
    first_sixty=indices[0:60]
    
    #Getting the column names of the selected features
    column_names=train_removed.columns
    final_columns=column_names[first_sixty]
    
    
    #Getting the final feature matrix
    final_feature_matrix=np.zeros([7848,60])
    for i in range(0,7848): 
        df=train_removed.iloc[[i]]
        final_feature_matrix[i,:]=df[final_columns].values
        
elif FS_final_strategie=='pca':        
    #-------------------------------OPTION 2: PCA----------------------------------
    pca = PCA(n_components=60, svd_solver='full')        
    final_feature_matrix=pca.fit_transform(train_removed)

# %% 
'''Saving the final feature set in a pickle'''
with open(r'wavelets_features_all_decomps_no_nans_variance_correlation_pca60.pickle', 'wb') as handle:
    pickle.dump(final_feature_matrix,handle,protocol=pickle.HIGHEST_PROTOCOL) 
