#Fonte: https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
# %%
import pickle
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
import warnings
warnings.simplefilter('ignore')
# %% SELECT FINAL FEATURE SELECTION STRATEGIE
#FS_final_strategie='pca'
FS_final_strategie='feature_importance'
# %%
with open(r'C:\Users\maria\Desktop\plasticc-kit-master\wavelets_features_all_decomps.pickle', 'rb') as handle:
    feature_matrix=pickle.load(handle)


#Removing nans and infs
feature_matrix_without_nans=OrderedDict()
for i in range(0,7848):
    print(i)
    df=feature_matrix[i]
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna(axis='columns')
    feature_matrix_without_nans[i]=df

#Getting final features

#Step 1: getting the columns in common
df=feature_matrix_without_nans[0]    
col_list=df.columns    
for i in range(1,7848):    
    print(i)
    df=feature_matrix_without_nans[i]  
    col_list=set(col_list) & set(df.columns)
      
#Getting feature matrix ready for pca
feature_matrix=np.zeros([7848,len(col_list)])
col_list=list(col_list)
for i in range(0,7848): 
    print(i)
    df=feature_matrix_without_nans[i]  
    #for ind in range (0,len(col_list)):
    feature_matrix[i,:]=df[col_list].values
    
 # %% FEATURE SELECTION
from sklearn.ensemble import ExtraTreesClassifier
meta_data = pd.read_csv('C:/Users/maria/Desktop/plasticc-kit-master/data/training_set_metadata.csv')
target_vec = meta_data['target']

# %% 
#---------------------REMOVING FEATURES WITH LOW VARIANCE----------------------
from sklearn.feature_selection import VarianceThreshold
features = pd.DataFrame(data=feature_matrix)
features=pd.concat([features, meta_data.iloc[:,[1,2,3,4,5,6,7,8,10]]], axis=1)

sel = VarianceThreshold(threshold=(.8 * (1 - .8))) #  remove all features that are either one or zero (on or off) in more than 80% of the samples
features=sel.fit_transform(features)
# %%
#------------------REMOVING FEATURES WITH HIGH CORRELATION---------------------
sys.path.append(r'C:\Users\maria\Desktop\plasticc-kit-master')
from feature_selector import FeatureSelector

# Features are in train and labels are in train_labels
features = pd.DataFrame(data=features)
fs = FeatureSelector(data = features, labels = target_vec)
fs.identify_collinear(correlation_threshold = 0.98)
fs.plot_collinear()

# list of collinear features to remove
collinear_features = fs.ops['collinear']
# dataframe of collinear features
fs.record_collinear.head()

# Remove the features from all methods (returns a df)
train_removed = fs.remove(methods = 'all')
# %% 
if FS_final_strategie=='feature_importance':
    #--------------------------OPTION 1: FEATURE IMPORTANCE------------------------
    # feature extraction
    model = ExtraTreesClassifier()
    model.fit(train_removed, target_vec)
    importances=model.feature_importances_
    #print(model.feature_importances_)
    
    indices = np.argsort(importances)[::-1]
    print(len(indices))
    
    first_sixty=indices[0:60]
    #Get column names
    column_names=train_removed.columns
    final_columns=column_names[first_sixty]
    
    
    #Final feature matrix
    final_feature_matrix=np.zeros([7848,60])
    
    # Here we have new operators, .iloc to explicity support only integer indexing, 
    #and .loc to explicity support only label indexing
    
    for i in range(0,7848): 
        df=train_removed.iloc[[i]]
        final_feature_matrix[i,:]=df[final_columns].values
        
elif FS_final_strategie=='pca':        
    #-------------------------------OPTION 2: PCA----------------------------------
    pca = PCA(n_components=60, svd_solver='full')        
    final_feature_matrix=pca.fit_transform(train_removed)

# %% PICKLE DUMPS
with open(r'wavelets_features_all_decomps_no_nans_variance_correlation_feature_importance60.pickle', 'wb') as handle:
    pickle.dump(final_feature_matrix,handle,protocol=pickle.HIGHEST_PROTOCOL)            








   