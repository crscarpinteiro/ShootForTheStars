import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import itertools
from sklearn.model_selection import StratifiedKFold
import pickle
from random import choice


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
    classes = sorted(np.unique(y_true))
    if len(classes) == 9:
        class_weight = {
            c: 1 for c in classes
        }
        for c in [0, 4]:
            class_weight[c] = 2
    else:
        class_weight = {
            c: 1 for c in classes
        }

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
    classes = sorted(np.unique(y_true))
    if len(classes) == 9:
        class_weight = {
            c: 1 for c in classes
        }
        for c in [0, 4]:
            class_weight[c] = 2
    else:
        class_weight = {
            c: 1 for c in classes
        }

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


def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_


def lgbm_modeling_cross_validation(params, full_train, y, nr_fold=5, random_state=1):
    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgb_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(fold_ + 1,
                                           multi_weighted_logloss(val_y, oof_preds[val_, :])))

        imp_df = pd.DataFrame({
            'feature': full_train.columns,
            'gain': clf.feature_importances_,
            'fold': [fold_ + 1] * len(full_train.columns),
        })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    df_importances = save_importances(importances_=importances)
    df_importances.to_csv('lgbm_importances.csv', index=False)

    return clfs, score


meta_data = pd.read_csv('training_set_metadata.csv')
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

# Read features
with open('different_n.pickle', 'rb') as f:
    features = pickle.load(f)

features = features[3]
# convert to Dataframe
features = pd.DataFrame(data=features)
features=pd.concat([features, meta_data.iloc[:,1:11]], axis=1)


galactic_features = features[galactic_cut]
extragalactic_features = features[~galactic_cut]

# map labels
targets_galactic = np.hstack([sorted(np.unique(galactic_target_vec))])
target_map = {j: i for i, j in enumerate(targets_galactic)}
target_ids = [target_map[i] for i in galactic_target_vec]
galactic_target_vec = pd.DataFrame(data=target_ids)

targets_extragalactic = np.hstack([sorted(np.unique(extragalactic_target_vec))])
target_map = {j: i for i, j in enumerate(targets_extragalactic)}
target_ids = [target_map[i] for i in extragalactic_target_vec]
extragalactic_target_vec = pd.DataFrame(data=target_ids)

# EXTRAGALACTIC
x_train_extragalactic, x_test_extragalactic, y_train_extragalactic, y_test_extragalactic = train_test_split(
    extragalactic_features, extragalactic_target_vec, test_size=0.2, random_state=0)

# scalling
scaler_extragalactic = StandardScaler().fit(x_train_extragalactic)
x_train_extragalactic = scaler_extragalactic.transform(x_train_extragalactic)
x_train_extragalactic = pd.DataFrame(data=x_train_extragalactic)
clfs = []
scores = []

# FIND BEST PARAMETERS
for i in range(150):
    params = {
        'objective': 'multiclass',
        'boosting_type': choice(['gbdt', 'dart']),
        'num_class': 9,
        'metric': 'multi_logloss',
        'subsample': .93,
        'colsample_bytree': .75,
        'reg_alpha': .01,
        'reg_lambda': .01,
        'min_split_gain': 0.01,
        'min_child_weight': 10,
        'verbosity': -1,
        'silent': True,
        'nthread': -1,
        'num_leaves': choice(np.linspace(5, 50, num=10)).astype(int),
        'n_estimators': choice(np.linspace(2000, 6000, num=100)).astype(int),
        'max_depth': choice(np.linspace(1, 10, num=10)).astype(int),
        'learning_rate': choice(np.linspace(0.001, 0.1, num=100)),
        'min_data_in_leaf': choice(np.linspace(15, 50, num=30)).astype(int),
        'bagging_freq': 10,
        'bagging_fraction': 0.99,
        'n_jobs': -1

    }

    print(i)
    this_clf, this_score = lgbm_modeling_cross_validation(params, x_train_extragalactic, y_train_extragalactic.iloc[:, 0],
                                                          nr_fold=5, random_state=1)
    clfs.append(this_clf)
    scores.append(this_score)

x_test_extragalactic = scaler_extragalactic.transform(x_test_extragalactic)
x_test_extragalactic = pd.DataFrame(data=x_test_extragalactic)

index = np.argmin(scores)
print('Best score for extra galactic:', np.min(scores))
best_clfs_extragalactic = clfs[index]
print(best_clfs_extragalactic[1].get_params())

# MAKE PREDICTIONS
preds = None
for clf in best_clfs_extragalactic:
    if preds is None:
        preds = clf.predict_proba(x_test_extragalactic[x_train_extragalactic.columns]) / len(best_clfs_extragalactic)
    else:
        preds += clf.predict_proba(x_test_extragalactic[x_train_extragalactic.columns]) / len(best_clfs_extragalactic)

y_pred_labels = []
for i in range(preds.shape[0]):
    y_pred_labels.append(np.argmax(preds[i]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_extragalactic, y_pred_labels)
np.set_printoptions(precision=9)
class_names = np.array(['15', '42', '52', '62', '64', '67', '88', '90', '95'])
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix,Extragalactic')
plt.show()

# GALACTIC
x_train_galactic, x_test_galactic, y_train_galactic, y_test_galactic = train_test_split(
    galactic_features, galactic_target_vec, test_size=0.2, random_state=0)
# # scalling
scaler_galactic = StandardScaler().fit(x_train_galactic)
x_train_galactic = scaler_galactic.transform(x_train_galactic)
x_train_galactic = pd.DataFrame(data=x_train_galactic)
clfs_galactic = []
scores=[]
for i in range(150):
    print(i)
    params = {
            'objective': 'multiclass',
            'boosting_type': choice(['gbdt', 'dart']),
            'num_class': 5,
            'metric': 'multi_logloss',
            'subsample': .93,
            'colsample_bytree': .75,
            'reg_alpha': .01,
            'reg_lambda': .01,
            'min_split_gain': 0.01,
            'min_child_weight': 10,
            'verbosity': -1,
            'silent': True,
            'nthread': -1,
            'num_leaves': choice(np.linspace(5, 50, num=10)).astype(int),
            'n_estimators': choice(np.linspace(2000, 6000, num=100)).astype(int),
            'max_depth': choice(np.linspace(1, 10, num=10)).astype(int),
            'learning_rate': choice(np.linspace(0.001, 0.1, num=100)),
            'min_data_in_leaf': choice(np.linspace(15, 50, num=30)).astype(int),
            'bagging_freq': 10,
            'bagging_fraction': 0.99,
            'n_jobs': -1

        }
    this_clf, this_score = lgbm_modeling_cross_validation(params, x_train_galactic, y_train_galactic.ix[:, 0],
                                                          nr_fold=5, random_state=1)
    clfs_galactic.append(this_clf)
    scores.append(this_score)

x_test_galactic = scaler_galactic.transform(x_test_galactic)
x_test_galactic = pd.DataFrame(data=x_test_galactic)

index = np.argmin(scores)
print('Best score for extra galactic:', np.min(scores))
best_clfs_galactic = clfs_galactic[index]
print(best_clfs_galactic[1].get_params())

# MAKE PREDICTIONS
preds = None
for clf in best_clfs_galactic:
    if preds is None:
        preds = clf.predict_proba(x_test_galactic[x_train_galactic.columns]) /len(best_clfs_galactic)
    else:
        preds += clf.predict_proba(x_test_galactic[x_train_galactic.columns]) / len(best_clfs_galactic)

y_pred_labels = []
for i in range(preds.shape[0]):
    y_pred_labels.append(np.argmax(preds[i]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_galactic, y_pred_labels)
np.set_printoptions(precision=5)
class_names = np.array(['6', '16', '53', '65', '92'])
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, Galactic')
plt.show()










# train one model for galactic and extragalactic objects

# Extragalactic Training


# params_extragalactic = {'boosting_type': 'gbdt',
#                         'objective': 'multiclass',
#                         'num_class': 9,
#                         'metric': 'multi_logloss',
#                         'subsample': .93,
#                         'colsample_bytree': .75,
#                         'reg_alpha': .01,
#                         'reg_lambda': .01,
#                         'min_split_gain': 0.01,
#                         'min_child_weight': 10,
#                         'verbosity': -1,
#                         'silent': True,
#                         'nthread': -1,
#                         'num_leaves': 32,
#                         'is_unbalanced': True,
#                         }
#
# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# clfs = []
# importances = pd.DataFrame()
#
# oof_preds = np.zeros((len(x_train_extragalactic), len(extragalactic_classes)))
# for fold_, (trn_, val_) in enumerate(folds.split(y_train_extragalactic, y_train_extragalactic)):
#     trn_x, trn_y = x_train_extragalactic.iloc[trn_], y_train_extragalactic.iloc[trn_]
#     val_x, val_y = x_train_extragalactic.iloc[val_], y_train_extragalactic.iloc[val_]
#
#     clf = lgb.LGBMClassifier(**params_extragalactic, learning_rate=0.01,
#                              n_estimators=1500, max_depth=4)
#     clf.fit(trn_x, trn_y,
#             eval_set=[(trn_x, trn_y), (val_x, val_y)],
#             eval_metric=lgb_multi_weighted_logloss,
#             verbose=100,
#             early_stopping_rounds=50
#             )
#     oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
#     print(multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))
#
#     imp_df = pd.DataFrame()
#     imp_df['feature'] = x_train_extragalactic.columns
#     imp_df['gain'] = clf.feature_importances_
#     imp_df['fold'] = fold_ + 1
#     importances = pd.concat([importances, imp_df], axis=0, sort=False)
#
#     clfs.append(clf)
#
# print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y_train_extragalactic, y_preds=oof_preds))
#
# mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
# importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
#
# plt.figure(figsize=(8, 12))
# sns.barplot(x='feature', y='gain', data=importances.sort_values('mean_gain', ascending=False))
# plt.tight_layout()
# plt.savefig('importances.png')
#
# # Make predictions
# x_test_extragalactic = scaler_extragalactic.transform(x_test_extragalactic)
# x_test_extragalactic = pd.DataFrame(data=x_test_extragalactic)
# preds = None
#
# for clf in clfs:
#     if preds is None:
#         preds = clf.predict_proba(x_test_extragalactic[x_train_extragalactic.columns]) / folds.n_splits
#     else:
#         preds += clf.predict_proba(x_test_extragalactic[x_train_extragalactic.columns]) / folds.n_splits
#     # Store predictions
#     preds_df = pd.DataFrame(preds, columns=['class_' + str(s) for s in clfs[0].classes_])
#
# y_pred_labels = []
# for i in range(preds.shape[0]):
#     y_pred_labels.append(np.argmax(preds[i]))
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test_extragalactic, y_pred_labels)
# np.set_printoptions(precision=9)
# class_names = np.array(['15', '42', '52','62', '64', '67', '88', '90','95'])
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Confusion matrix,Extragalactic')
# plt.show()
#
# # Galactic Training
#
# x_train_galactic, x_test_galactic, y_train_galactic, y_test_galactic = train_test_split(
#     galactic_features, galactic_target_vec, test_size=0.2, random_state=0)
#
# # scalling
# scaler_galactic = StandardScaler().fit(x_train_galactic)
# x_train_galactic = scaler_galactic.transform(x_train_galactic)
# x_train_galactic = pd.DataFrame(data=x_train_galactic)
# params_galactic = {
#     'objective': 'multiclass',
#     'boosting_type': 'gbdt',
#     'num_class': 5,
#     'metric': 'multi_logloss',
#     'subsample': .93,
#     'colsample_bytree': .75,
#     'reg_alpha': .01,
#     'reg_lambda': .01,
#     'min_split_gain': 0.01,
#     'min_child_weight': 10,
#     'verbosity': -1,
#     'silent': True,
#     'is_unbalanced': True,
#     'nthread': -1,
#     'num_leaves': 16
# }
#
# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# clfs = []
# importances = pd.DataFrame()
#
# oof_preds = np.zeros((len(x_train_galactic), len(galactic_classes)))
# for fold_, (trn_, val_) in enumerate(folds.split(y_train_galactic, y_train_galactic)):
#     trn_x, trn_y = x_train_galactic.iloc[trn_], y_train_galactic.iloc[trn_]
#     val_x, val_y = x_train_galactic.iloc[val_], y_train_galactic.iloc[val_]
#
#     clf = lgb.LGBMClassifier(**params_galactic,
#                              n_estimators=1500, learning_rate=0.005, max_depth=4)
#     clf.fit(trn_x, trn_y,
#             eval_set=[(trn_x, trn_y), (val_x, val_y)],
#             eval_metric=lgb_multi_weighted_logloss,
#             verbose=100,
#             early_stopping_rounds=50
#             )
#     oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
#     print(multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))
#
#     imp_df = pd.DataFrame()
#     imp_df['feature'] = x_train_galactic.columns
#     imp_df['gain'] = clf.feature_importances_
#     imp_df['fold'] = fold_ + 1
#     importances = pd.concat([importances, imp_df], axis=0, sort=False)
#
#     clfs.append(clf)
#
# print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y_train_galactic, y_preds=oof_preds))
#
# mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
# importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
#
# plt.figure(figsize=(8, 12))
# sns.barplot(x='feature', y='gain', data=importances.sort_values('mean_gain', ascending=False))
# plt.tight_layout()
# plt.savefig('importances.png')
#
# # Make predictions
# x_test_galactic = scaler_galactic.transform(x_test_galactic)
# x_test_galactic = pd.DataFrame(data=x_test_galactic)
# preds = None
# for clf in clfs:
#     if preds is None:
#         preds = clf.predict_proba(x_test_galactic[x_train_extragalactic.columns]) / folds.n_splits
#     else:
#         preds += clf.predict_proba(x_test_galactic[x_train_extragalactic.columns]) / folds.n_splits
#     # Store predictions
#     preds_df = pd.DataFrame(preds, columns=['class_' + str(s) for s in clfs[0].classes_])
#
# y_pred_labels = []
# for i in range(preds.shape[0]):
#     y_pred_labels.append(np.argmax(preds[i]))
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test_galactic, y_pred_labels)
# np.set_printoptions(precision=5)
# class_names = np.array(['6', '16', '53', '65', '92'])
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Confusion matrix, Galactic')
# plt.show()
