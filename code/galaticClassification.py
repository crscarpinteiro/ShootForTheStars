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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    class_weight = {0: 2, 1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}

    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    class_weight = {0: 2, 1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


meta_data = pd.read_csv('training_set_metadata.csv')
targets = np.hstack([np.unique(meta_data['target']), [99]])  # 99 refers to the class that doesn't show in training
target_vec = meta_data['target']

# Divide between extragalatic and galatic classes
galactic_bool = meta_data['hostgal_photoz'] == 0
exact_bool = meta_data['hostgal_photoz'].notnull().astype(int)
galactic_classes = np.sort(meta_data.loc[meta_data['target'].notnull() & galactic_bool, 'target'].unique().astype(int))
extragalactic_classes = np.sort(
    meta_data.loc[meta_data['target'].notnull() & ~galactic_bool, 'target'].unique().astype(int))

# Build the arrays for both the galactic and extragalactic groups
galactic_cut = meta_data['hostgal_specz'] == 0
galactic_data = meta_data[galactic_cut]
extragalactic_data = meta_data[~galactic_cut]

galactic_target_vec = galactic_data['target'].values
extragalactic_target_vec = extragalactic_data['target'].values

# map labels to 0 to 14
target_map = {j: i for i, j in enumerate(targets)}
target_ids = [target_map[i] for i in meta_data['target']]
meta_data['target_id'] = target_ids

# Read features
feature_set_file = open('feature_set.npy', 'rb')
features = np.load(feature_set_file)
galactic_features = features[galactic_cut]
extragalactic_features = features[~galactic_cut]

# map labels to 0 to 14
targets_galactic = np.hstack([np.unique(galactic_target_vec)])
target_map = {j: i for i, j in enumerate(targets_galactic)}
target_ids = [target_map[i] for i in galactic_target_vec]
galactic_target_vec = target_ids

targets_extragalactic = np.hstack([np.unique(extragalactic_target_vec)])
target_map = {j: i for i, j in enumerate(targets_extragalactic)}
target_ids = [target_map[i] for i in extragalactic_target_vec]
extragalactic_target_vec = target_ids

# train one model for each class

x_train_extragalactic, x_test_extragalactic, y_train_extragalactic, y_test_extragalactic = train_test_split(
    extragalactic_features,
    extragalactic_target_vec,
    test_size=0.2, random_state=0)
ss_extragalactic = StandardScaler().fit(x_train_extragalactic)
x_train_extragalactic = ss_extragalactic.transform(x_train_extragalactic)
x_test_extragalactic = ss_extragalactic.transform(x_test_extragalactic)

# Galactic training
num_rounds = 3000

d_train_extragalactic = lgb.Dataset(x_train_extragalactic, label=y_train_extragalactic)
d_test_extragalactic = lgb.Dataset(x_test_extragalactic, label=y_test_extragalactic)
params_extragalactic = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'subsample': .93,
    'colsample_bytree': .75,
    'reg_alpha': .01,
    'reg_lambda': .01,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'verbosity': -1,
    'silent': True,
    'nthread': -1
}

learning_rate = 0.05
n_estimators = 1500
max_depth = 4

clf = lgb.LGBMClassifier(**params_extragalactic, learning_rate=learning_rate,
                         n_estimators=n_estimators, max_depth=max_depth)

clf.fit(x_train_extragalactic, np.asarray(y_train_extragalactic),
        eval_set=[(x_train_extragalactic, np.asarray(y_train_extragalactic).reshape((len(np.asarray(y_train_extragalactic)), 1))),
                  (x_test_extragalactic, np.asarray(y_test_extragalactic).reshape((len(np.asarray(y_test_extragalactic)), 1)))],
        verbose=False, early_stopping_rounds=50, eval_metric=lgb_multi_weighted_logloss)

oof_preds = clf.predict_proba(x_test_extragalactic, num_iteration=clf.best_iteration_)

loss = multi_weighted_logloss(y_true=y_test_extragalactic, y_preds=oof_preds)
y_pred_labels=[]
print('MULTI WEIGHTED LOG LOSS : %.5f ' % loss)
for i in range(oof_preds.shape[0]):
    y_pred_labels.append(np.argmax(oof_preds[i]))

acc = sklearn.metrics.accuracy_score(y_test_extragalactic, y_pred_labels, normalize=True, sample_weight=None)
print('accuracy in the test set:', acc)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_extragalactic, y_pred_labels)
np.set_printoptions(precision=9)
class_names = np.array(['1', '2', '3', '4', '5','6','7','8'])
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()

#
# x_train_galactic, x_test_galactic, y_train_galactic, y_test_galactic = train_test_split(galactic_features,
#                                                                                         galactic_target_vec,
#                                                                                         test_size=0.2, random_state=0)
# ss_galactic = StandardScaler().fit(x_train_galactic)
# x_train_galactic = ss_galactic.transform(x_train_galactic)
# x_test_galactic = ss_galactic.transform(x_test_galactic)

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test_extragalactic, y_pred_labels)
# np.set_printoptions(precision=9)
# class_names = np.array(['15', '42', '52', '62', '64', '67', '88', '90', '95'])
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Confusion matrix, with normalization')
# plt.show()
#
# # Galactic training
# num_rounds = 5000
#
# d_train_galactic = lgb.Dataset(x_train_galactic, label=y_train_galactic)
# d_test_galactic = lgb.Dataset(x_test_galactic, label=y_test_galactic)
#
# params_galactic = {'boosting_type': 'gbdt', 'objective': 'multiclass_ova','num_class': 5,
#                    'num_leaves': 16, 'seed': 0, 'verbose': -1, 'min_data_in_leaf': 1, 'bagging_fraction': 0.8,
#                    'bagging_freq': 1, 'lambda_l1': 0, 'lambda_l2': 1, 'metric':'multi_logloss', 'learning_rate': 0.05}
#
# # estimator
# model = lgb.train(params=params_galactic, train_set=d_train_galactic)
#
# y_pred_prob = model.predict(x_test_galactic)
# y_pred_labels = []
# for i in range(y_pred_prob.shape[0]):
#     y_pred_labels.append(np.argmax(y_pred_prob[i]))
#
# acc = sklearn.metrics.accuracy_score(y_test_galactic, y_pred_labels, normalize=True, sample_weight=None)
# print('accuracy in the test set:', acc)
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test_galactic, y_pred_labels)
# np.set_printoptions(precision=5)
# class_names = np.array(['6', '16', '53', '65', '92'])
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Confusion matrix, with normalization')
# plt.show()
