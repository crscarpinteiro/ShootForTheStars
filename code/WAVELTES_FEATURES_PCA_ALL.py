'''
This script decomposes th data using the wavelet transform and extract features upon its resulr
'''
# %%
#Importing libraries
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
# %%
'''Reading the data'''
filename='C:/Users/maria/Desktop/plasticc-kit-master/data/training_set.csv'
training_data = Table.read(filename, format='ascii.csv')

'''Reading the metadata'''
filename='C:/Users/maria/Desktop/plasticc-kit-master/data/training_set_metadata.csv'
meta_training = Table.read(filename, format='ascii.csv')
nobjects = len(meta_training)

'''Building a dictionary of the light bands'''
pbmap = OrderedDict([(0,'u'), (1,'g'), (2,'r'), (3,'i'), (4, 'z'), (5, 'y')])

'''Associating each light band with a color, to facilitate plots later on'''
pbcols = OrderedDict([(0,'blueviolet'), (1,'green'), (2,'red'),\
                      (3,'orange'), (4, 'black'), (5, 'brown')])
pbnames = list(pbmap.values())
# %%
'''List of the features which will be extracted using Cesium'''
features_to_use = ["all_times_nhist_numpeaks",
                  "all_times_nhist_peak1_bin",
                  "all_times_nhist_peak2_bin",
                  "all_times_nhist_peak3_bin",
                  "all_times_nhist_peak4_bin",
                  "all_times_nhist_peak_1_to_2",
                  "all_times_nhist_peak_1_to_3",
                  "all_times_nhist_peak_1_to_4",
                  "all_times_nhist_peak_2_to_3",
                  "all_times_nhist_peak_2_to_4",
                  "all_times_nhist_peak_3_to_4",
                  "all_times_nhist_peak_val",
                  "avg_double_to_single_step",
                  "avg_err",
                  "avgt",
                  "cad_probs_1",
                  "cad_probs_10",
                  "cad_probs_20",
                  "cad_probs_30",
                  "cad_probs_40",
                  "cad_probs_50",
                  "cad_probs_100",
                  "cad_probs_500",
                  "cad_probs_1000",
                  "cad_probs_5000",
                  "cad_probs_10000",
                  "cad_probs_50000",
                  "cad_probs_100000",
                  "cad_probs_500000",
                  "cad_probs_1000000",
                  "cad_probs_5000000",
                  "cad_probs_10000000",
                  "cads_avg",
                  "cads_med",
                  "cads_std",
                  "mean",
                  "med_double_to_single_step",
                  "med_err",
                  "n_epochs",
                  "std_double_to_single_step",
                  "std_err",
                  "total_time",
                  "amplitude",
                  "flux_percentile_ratio_mid20",
                  "flux_percentile_ratio_mid35",
                  "flux_percentile_ratio_mid50",
                  "flux_percentile_ratio_mid65",
                  "flux_percentile_ratio_mid80",
                  "max_slope",
                  "maximum",
                  "median",
                  "median_absolute_deviation",
                  "minimum",
                  "percent_amplitude",
                  "percent_beyond_1_std",
                  "percent_close_to_median",
                  "percent_difference_flux_percentile",
                  "period_fast",
                  "qso_log_chi2_qsonu",
                  "qso_log_chi2nuNULL_chi2nu",
                  "skew",
                  "std",
                  "stetson_j",
                  "stetson_k",
                  "weighted_average",
                  "fold2P_slope_10percentile",
                  "fold2P_slope_90percentile",
                  "freq1_amplitude1",
                  "freq1_amplitude2",
                  "freq1_amplitude3",
                  "freq1_amplitude4",
                  "freq1_freq",
                  "freq1_lambda",
                  "freq1_rel_phase2",
                  "freq1_rel_phase3",
                  "freq1_rel_phase4",
                  "freq1_signif",
                  "freq2_amplitude1",
                  "freq2_amplitude2",
                  "freq2_amplitude3",
                  "freq2_amplitude4",
                  "freq2_freq",
                  "freq2_rel_phase2",
                  "freq2_rel_phase3",
                  "freq2_rel_phase4",
                  "freq3_amplitude1",
                  "freq3_amplitude2",
                  "freq3_amplitude3",
                  "freq3_amplitude4",
                  "freq3_freq",
                  "freq3_rel_phase2",
                  "freq3_rel_phase3",
                  "freq3_rel_phase4",
                  "freq_amplitude_ratio_21",
                  "freq_amplitude_ratio_31",
                  "freq_frequency_ratio_21",
                  "freq_frequency_ratio_31",
                  "freq_model_max_delta_mags",
                  "freq_model_min_delta_mags",
                  "freq_model_phi1_phi2",
                  "freq_n_alias",
                  "freq_signif_ratio_21",
                  "freq_signif_ratio_31",
                  "freq_varrat",
                  "freq_y_offset",
                  "linear_trend",
                  "medperc90_2p_p",
                  "p2p_scatter_2praw",
                  "p2p_scatter_over_mad",
                  "p2p_scatter_pfold_over_mad",
                  "p2p_ssqr_diff_over_var",
                  "scatter_res_raw"]
# %%
feature_matrix=OrderedDict() #variable which will contain the feature matrix

wavelet_decomp=OrderedDict() #variable which will contain the wavelet decomposition
w = pywt.Wavelet('sym2') #inicialization of the selected wavelet

for i in tnrange(nobjects, desc='Building Timeseries'):
          times=[]
          measures=[]
          print(i)
          row = meta_training[i]
          thisid = row['object_id']
          target = row['target']
    
          meta = {'z':row['hostgal_photoz'],\
                  'zerr':row['hostgal_photoz_err'],\
                  'mwebv':row['mwebv']}
    
          ind = (training_data['object_id'] == thisid)
          thislc = training_data[ind]
    
          pbind = [(thislc['passband'] == pb) for pb in pbmap]
          t = [thislc['mjd'][mask].data for mask in pbind ]# t = tempo
          measures = [thislc['flux'][mask].data for mask in pbind ] # m = measurement
          e = [thislc['flux_err'][mask].data for mask in pbind ]
          
          
          max_len=100        
          for ind in range(0,6): 
              while len(measures[ind])!=max_len:
                  measures[ind]=np.append(measures[ind],0)

          coef_list_for_obj=np.zeros([1,max_len])
                  
          wavelet_decomp[thisid]=[pywt.swt(m, w, level=2) for m in measures] #wavelet decomposition
          current_wavelet_decom=wavelet_decomp[thisid]    
          for a in range (0,6):
              this_band=current_wavelet_decom[a]
              level1=this_band[0]
              level2=this_band[1]
              l1_approx=level1[0]
              l1_detail=level1[1]
              l2_approx=level2[0]
              l2_detail=level2[1]
              coef_list_for_band=np.vstack((l1_detail,l2_detail))
              coef_list_for_obj=np.vstack((coef_list_for_obj,coef_list_for_band))
          coef_list_for_obj=coef_list_for_obj[1:,:]
          
          #Extracting features from the set of bands resulting from the wavelet decomposition
          feature_matrix[i] = featurize.featurize_time_series(times=None, values=coef_list_for_obj,  features_to_use=features_to_use, meta_features=None)              

# %%
#Saving the features into a pickle file
with open(r'wavelets_features_details_decomps.pickle', 'wb') as handle:
    pickle.dump(feature_matrix,handle,protocol=pickle.HIGHEST_PROTOCOL)  

