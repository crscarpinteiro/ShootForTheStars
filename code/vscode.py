import os
import numpy as np
import scipy.stats as spstat
import matplotlib.pyplot as plt
from collections import OrderedDict
import sncosmo
from astropy.table import Table
import pywt
import pywt.data
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plot
import sys
import os
from collections import Counter, OrderedDict
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from astropy.table import Table
import schwimmbad
from cesium.time_series import TimeSeries
import cesium.featurize as featurize
from tqdm import tnrange, tqdm_notebook
import sklearn 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pickle

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

#space_times = np.load("st.npy")
#space_measurements = np.load('meas.npy')
#space_meta_features = np.load('meta.npy')
print(len(space_times))

#space_times.tolist()
#space_measurements.tolist()
space_meta_features.tolist()

print(len(space_times))

features_interp = []
for index in range(len(space_meta_features)):
    print(index)
    fset = featurize.featurize_time_series(times=space_times[index], values=space_measurements[index], errors=None, features_to_use=features_to_use, meta_features=space_meta_features[index])
    features_interp.append(fset)
    
print(len(features_interp))


f = open('features_interp.p', 'wb')   # 'wb' instead 'w' for binary file
pickle.dump(features_interp, f, -1)       # -1 specifies highest binary protocol
f.close()