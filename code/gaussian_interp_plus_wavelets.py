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


np.random.seed(1)

#Alguns kernels para o Gaussian Regression Process
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernel1 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) 
kernel2=1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
kernel3=1.0 * RationalQuadratic(length_scale=1.0, alpha=1) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
kernel6=1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)

#Parâmetro nu
'''
The smaller nu, the less smooth the approximated function is.
For nu=inf, the kernel becomes equivalent to the RBF kernel and for nu=0.5 to 
the absolute exponential kernel. Important intermediate values are nu=1.5 
(once differentiable functions) and nu=2.5 (twice differentiable functions).
'''

#Parâmetro lengthscale
'''
The length scale of the kernel. If a float, an isotropic kernel is used.
If an array, an anisotropic kernel is used where each dimension of l defines 
the length-scale of the respective feature dimension.
'''

#Segundo o vencedor do challenge no kaggle:
'''
I first use Gaussian process (GP) regression to extract features. 
I trained a GP on each object using a Matern Kernel with a fixed length scale
in the wavelength direction and a variable length scale in the time direction.
'''

#Tentativa de replicar o kernel que o vencedro usou:
winner_kernel=Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=2.5)

# Criar o objeto wavelet
w = pywt.Wavelet('sym2')

gp = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=15) #o kernel escolhido foi o RBF, o resultado é igual ao do Mattern, pelo menos visualmente
#Se quisermos escrver os valores previstos num csv, descomentar
#w = csv.writer(open("bands_matern.csv", "w"),delimiter=',')
#z = csv.writer(open("times_matern.csv", "w"),delimiter=',')
wavelet_decomp=OrderedDict()
features_interp = OrderedDict()
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
          measures = [thislc['flux'][mask].data for mask in pbind ] # m = measurement (flux)
          e = [thislc['flux_err'][mask].data for mask in pbind ] #e = flux error
        
          #plot_multicolor_lc(t,m)
        
          for a in range (0, 6):
                   this_t=t[a].reshape([len(t[a]),1])
                   this_m=measures[a].reshape([len(e[a]),1])
                   this_e=e[a].reshape([len(e[a]),1])  
        
                    #Gaussian process regression
                   index_points = np.linspace(this_t.min(), this_t.max(), 100)
                   index_points=index_points.reshape([len(index_points),1])

                   gp.fit(this_t, this_m)            
                   x = np.atleast_2d(np.linspace(this_t.min(), this_t.max(), 100)).T         
                   y_pred, sigma = gp.predict(x, return_std=True)
                   times.append(index_points)
                   measures.append(y_pred)
                   
                   #Se quisermos escrver os valores previstos num csv, descomentar
                   #w.writerow(y_pred)   
                   #z.writerow(index_points)
                   
                   #plot_true_plus_interp(this_t, this_m, index_points, y_pred, a)
          
            
          #Descomentar para obter as wavelets a partir da interpolação
          #wavelet_decomp[thisid]=[pywt.wavedec(m, w, level=2) for m in measures]          
          
          #plot_multicolor_lc(times, measures)
          
          #Descomentar para obter as features a partir da interpolação
          #features_interp[i] = featurize.featurize_time_series(times=times, values=measures, errors=None, features_to_use=features_to_use, meta_features=meta)
 
    
    


'''
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
              objs_measures.append(np.asarray(times[iterator]))
              objs_times.append(np.asarray(bands[iterator]))
              iterator=iterator+1
      
      ts_times[thisid] = objs_times
      ts_measures[thisid] = objs_measures
'''

###############################################################################
'''
with open('times_with_interp.pickle', 'rb') as handle:
    times_dict = pickle.load(handle)    
    
'''
###############################################################################
'''
À trous wavelet transform é esta!!!!!! https://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html
'''
###############################################################################
'''
"""Experiments with fake data"""

#Reading fake data
filename='C:/Users/maria/Desktop/plasticc-kit-master/data/fake010.csv'
fake = Table.read(filename, format='ascii.csv')

times=[]
measures=[]
row = fake[i]
thisid = row['object_id']
    
ind = (fake['object_id'] == thisid)
thislc = fake[ind]
    
pbind = [(thislc['passband'] == pb) for pb in pbmap]
t = [thislc['mjd'][mask].data for mask in pbind ]# t = tempo
m = [thislc['flux'][mask].data for mask in pbind ] # m = measurement
e = [thislc['flux_err'][mask].data for mask in pbind ]
        
plot_multicolor_lc(t,m)  
'''