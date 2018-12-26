import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter, OrderedDict
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from astropy.table import Table
import multiprocessing
from cesium.time_series import TimeSeries
import cesium.featurize as featurize
from tqdm import tnrange, tqdm_notebook
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

meta_data = pd.read_csv('training_set_metadata.csv')
targets = np.hstack([np.unique(meta_data['target']), [99]])  # 99 refers to the class that doesn't show in training

# map labels to 0 to 14
target_map = {j: i for i, j in enumerate(targets)}
target_ids = [target_map[i] for i in meta_data['target']]
meta_data['target_id'] = target_ids

# Build the flat probability arrays for both the galactic and extragalactic groups
galactic_cut = meta_data['hostgal_specz'] == 0
galactic_data = meta_data[galactic_cut]
extragalactic_data = meta_data[~galactic_cut]

galactic_classes = np.unique(galactic_data['target_id'])
extragalactic_classes = np.unique(extragalactic_data['target_id'])

# Add class 99 (id=14) to both groups.
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)

galactic_probabilities = np.zeros(15)
galactic_probabilities[galactic_classes] = 1. / len(galactic_classes)

extragalactic_probabilities = np.zeros(15)
extragalactic_probabilities[extragalactic_classes] = 1. / len(extragalactic_classes)

import tqdm


def do_prediction(table):
    probs = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
        else:
            prob = extragalactic_probabilities
        probs.append(prob)
    return np.array(probs)


pred = do_prediction(meta_data)

pbmap = OrderedDict([(0, 'u'), (1, 'g'), (2, 'r'), (3, 'i'), (4, 'z'), (5, 'Y')])

# it also helps to have passbands associated with a color
pbcols = OrderedDict([(0, 'blueviolet'), (1, 'green'), (2, 'red'), \
                      (3, 'orange'), (4, 'black'), (5, 'brown')])

pbnames = list(pbmap.values())

metafilename = 'training_set_metadata.csv'
metadata = Table.read(metafilename, format='csv')
nobjects = len(metadata)

lcfilename = 'training_set.csv'
lcdata = Table.read(lcfilename, format='csv')

tsdict = OrderedDict()
# for i in tnrange(nobjects, desc='Building Timeseries'):
#     row = metadata[i]
#     thisid = row['object_id']
#     target = row['target']
#
#     meta = {'z': row['hostgal_photoz'],
#             'zerr': row['hostgal_photoz_err'],
#             'mwebv': row['mwebv']}
#
#     ind = (lcdata['object_id'] == thisid)
#     thislc = lcdata[ind]
#
#     pbind = [(thislc['passband'] == pb) for pb in pbmap]
#     t = [thislc['mjd'][mask].data for mask in pbind]
#     m = [thislc['flux'][mask].data for mask in pbind]
#     e = [thislc['flux_err'][mask].data for mask in pbind]
#
#     tsdict[thisid] = TimeSeries(t=t, m=m, e=e,
#                                 label=target, name=thisid, meta_features=meta,
#                                 channel_names=pbnames)
#     print(i)
#
# del lcdata
#
# features_to_use = ["amplitude",
#                    "percent_beyond_1_std",
#                    "maximum",
#                    "max_slope",
#                    "median",
#                    "median_absolute_deviation",
#                    "percent_close_to_median",
#                    "minimum",
#                    "skew",
#                    "std",
#                    "weighted_average"]
#
# import warnings
# warnings.simplefilter('ignore')
#
# def worker(tsobj):
#     global features_to_use
#     thisfeats = featurize.featurize_single_ts(tsobj, features_to_use=features_to_use, raise_exceptions=False)
#     return thisfeats
#
#
# featurefile = f'plasticc_featuretable.npz'
# if os.path.exists(featurefile):
#     featuretable, _ = featurize.load_featureset(featurefile)
# else:
#     features_list = []
#     with tqdm_notebook(total=nobjects, desc="Computing Features") as pbar:
#         with multiprocessing.Pool() as pool:
#             results = pool.imap(worker, list(tsdict.values()))
#             for res in results:
#                 features_list.append(res)
#                 print(len(features_list))
#                 pbar.update()
#
#     featuretable = featurize.assemble_featureset(features_list=features_list, time_series=tsdict.values())
#     featurize.save_featureset(fset=featuretable, path=featurefile)

