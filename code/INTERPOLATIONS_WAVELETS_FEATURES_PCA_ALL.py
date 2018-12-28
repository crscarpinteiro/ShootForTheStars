'''
Ficheiros que obtemos com este código:
gp_wavelets_features_all_decomps -> feito
gp_wavelets_features_details_decomps ->   feito

gp_wavelets_features_all_decomps_no_nans_pca -> feito
gp_wavelets_features_details_decomps_no_nans_pca -> feito
'''
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
# %%
# %%
#Leitura dos dados
filename='C:/Users/maria/Desktop/plasticc-kit-master/data/training_set.csv'
training_data = Table.read(filename, format='ascii.csv')

#Leitura dos metadados
filename='C:/Users/maria/Desktop/plasticc-kit-master/data/training_set_metadata.csv'
meta_training = Table.read(filename, format='ascii.csv')
nobjects = len(meta_training)

print('Meta data')
print(meta_training.info)
print(meta_training)

"""Dict das bandas passantes"""

pbmap = OrderedDict([(0,'u'), (1,'g'), (2,'r'), (3,'i'), (4, 'z'), (5, 'y')])

#Associação de cada banda a uma cor, dará jeito no futuro para plots
pbcols = OrderedDict([(0,'blueviolet'), (1,'green'), (2,'red'),\
                      (3,'orange'), (4, 'black'), (5, 'brown')])

pbnames = list(pbmap.values())

"""Plot simultâneo de todas as bandas de uma observação"""

def plot_multicolor_lc(times, measures):

        fig, ax = plt.subplots(figsize=(8,6))

        xlabel = 'MJD'
            
        for i in range (0,6):
            pbname= pbcols[i]
            ax.errorbar(times[i], 
                     measures[i],
                     fmt = '*', color = pbname, label = f'{pbname}', markersize=5)
            
        ax.legend(ncol = 4, frameon = True)
        ax.set_xlabel(f'{xlabel}', fontsize='large')
        ax.set_ylabel('Flux', fontsize='large')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
 
def plot_true_plus_interp(times_original, measures_original, times_interp, measures_interp, i):

        fig, ax = plt.subplots(figsize=(8,2))

        xlabel = 'MJD'
            
        
        pbname= pbcols[i]
        ax.errorbar(times_original, 
                     measures_original,
                     fmt = '*', color = pbname, label = f'{pbname}', markersize=5)
        ax.errorbar(times_interp, 
                     measures_interp,
                     fmt = '-', color = pbname, label = f'{pbname}', markersize=5)
                        
        ax.legend(ncol = 4, frameon = True)
        ax.set_xlabel(f'{xlabel}', fontsize='large')
        ax.set_ylabel('Flux', fontsize='large')
        fig.tight_layout(rect=[0, 0, 1, 0.97])  



# Criar o objeto wavelet
w = pywt.Wavelet('sym2')
wavelet_decomp=OrderedDict()      
''' 
with open('wavelet_decomp_without_interp.pickle', 'rb') as handle:
    wavelet_decomp = pickle.load(handle)         

wavelet_decomp_interp=OrderedDict()     
with open('wavelet_decomp_with_interp.pickle', 'rb') as handle:
    wavelet_decomp_interp = pickle.load(handle) 
'''
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
# %%    WAVELET DECOMPOSITION WITH THE INTERPOLATIONS
#Código exemplo para a leitura dos csvs
filename='C:/Users/maria/Desktop/plasticc-kit-master/bands.csv'
bands=[]
with open(filename, newline='') as csvfile:
    reading = csv.reader(csvfile, delimiter=',')
    for row in reading:
        if row !=[]:
            bands.append(row)

filename='C:/Users/maria/Desktop/plasticc-kit-master/times.csv'
times=[]
with open(filename, newline='') as csvfile:
    reading = csv.reader(csvfile, delimiter=',')
    for row in reading:
        if row !=[]:
            times.append(row)

ts_times= OrderedDict()
ts_measures= OrderedDict()
iterator=0

for i in tnrange(nobjects, desc='Building Timeseries'):
      print(i)
      row = meta_training[i]
      thisid = row['object_id']
      target = row['target']
    
      meta = {'z':row['hostgal_photoz'],\
            'zerr':row['hostgal_photoz_err'],\
            'mwebv':row['mwebv']}
      objs_measures=[]
      objs_times=[]
      for a in range (0, 6):
              print(a)
              times_vec=[]
              measures_vec=[]
              this_times=times[iterator]
              this_measures=bands[iterator]
              for b in range (0, len(this_times)):
                  cenas=this_times[b]
                  str_value=cenas[1:len(cenas)-1]
                  float_value=float(str_value)
                  times_vec.append(float_value)
                  measures_vec.append(float_value)
                  
              objs_measures.append(np.asarray(times_vec))
              objs_times.append(np.asarray(measures_vec))
              iterator=iterator+1
      
      ts_times[thisid] = objs_times
      ts_measures[thisid] = objs_measures

wavelet_decomp=OrderedDict()
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
          
          current_interp=ts_measures[thisid]
          wavelet_decomp[thisid]=[pywt.swt(m, w, level=2) for m in current_interp]  

# %%
feature_matrix=OrderedDict() #O 10 deve-se às features dos metadados
for i in range(0,7848):
          times=[]
          measures=[]
          print(i)
          row = meta_training[i]
          thisid = row['object_id']
          target = row['target']

          ind = (training_data['object_id'] == thisid)
          thislc = training_data[ind]

          current_wavelet_decom=wavelet_decomp[thisid]
          coef_list_for_obj=np.zeros([1,100])
          for a in range (0,6):
              this_band=current_wavelet_decom[a]
              level1=this_band[0]
              level2=this_band[1]
              l1_approx=level1[0]
              l1_detail=level1[1]
              l2_approx=level2[0]
              l2_detail=level2[1]
              coef_list_for_band=np.vstack((l1_approx,l1_detail,l2_approx,l2_detail))
              coef_list_for_obj=np.vstack((coef_list_for_obj,coef_list_for_band))
          coef_list_for_obj=coef_list_for_obj[1:,:]
          feature_matrix[i] = featurize.featurize_time_series(times=None, values=coef_list_for_obj,  features_to_use=features_to_use, meta_features=None)              
# %%

with open(r'gp_wavelets_features_all_decomps.pickle', 'wb') as handle:
    pickle.dump(feature_matrix,handle,protocol=pickle.HIGHEST_PROTOCOL)  

# %%     

with open(r'C:\Users\maria\Desktop\plasticc-kit-master\gp_wavelets_features_all_decomps.pickle', 'rb') as handle:
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
feature_matrix_for_pca=np.zeros([7848,len(col_list)])
col_list=list(col_list)
for i in range(0,7848): 
    print(i)
    df=feature_matrix_without_nans[i]  
    #for ind in range (0,len(col_list)):
    feature_matrix_for_pca[i,:]=df[col_list].values

# %%
pca = PCA(n_components=60, svd_solver='full')        
new_features=pca.fit_transform(feature_matrix_for_pca)

with open(r'gp_wavelets_features_all_decomps_no_nans_pca.pickle', 'wb') as handle:
    pickle.dump(new_features,handle,protocol=pickle.HIGHEST_PROTOCOL)            