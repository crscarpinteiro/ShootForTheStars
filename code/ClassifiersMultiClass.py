from sklearn import neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
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
from sklearn.metrics import accuracy_score
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pickle
import pprint
import sys
from sklearn.metrics import confusion_matrix
import itertools

filename ='C:\\Users\\Cristiana\\Documents\\GitHub\\ShootForTheStars\\code\\training_set_metadata.csv'
meta_training = Table.read(filename, format='ascii.csv')
n_objects = len(meta_training)
target_vec = np.asarray(meta_training['target'])


# Read npy array of features
feature_set_file = open('C:\\Users\\Cristiana\\Documents\\GitHub\\ShootForTheStars\\code\\feature_set.npy', 'rb')
features = np.load(feature_set_file)
pprint.pprint(features)
plt.plot(features[:, 1])

x_train, x_test, y_train, y_test = train_test_split(features, target_vec, test_size=0.2, random_state=0)

# pca = PCA(n_components='mle', whiten=True, svd_solver="full", random_state=42)
# x_train_pca = pca.fit_transform(x_train)


def svm_classifier(x_train, y_train):
    svm_parameters = { 'C': [0.01, 0.5, 1, 10, 100]}
    grid_svm = RandomizedSearchCV(svm.LinearSVC(class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0),svm_parameters,
                                  scoring='accuracy', cv=5, n_jobs=-1, verbose=3)

    grid_svm.fit(x_train, y_train)
    print('Best SVM Parameters:')
    print(grid_svm.best_params_)
    svm_score = grid_svm.best_score_
    final_svm_model = grid_svm

    return svm_score, final_svm_model


def knn_classifier(x_train, y_train):
    n_neighbors = [1, 12, 20, 32, 37, 40, 50]

    index = -1
    maximum = 0

    for n in range(len(n_neighbors)):
        nr_neighs = n_neighbors[n]
        knn = KNeighborsClassifier(nr_neighs)
        knn_cv_scores = cross_val_score(
            knn, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        knn_score = knn_cv_scores.mean()

        if knn_score > maximum:
            maximum = knn_score
            index = nr_neighs

    knn_final = KNeighborsClassifier(index)
    knn_final.fit(x_train, y_train)

    return maximum, knn_final


def bayes_classifier(x_train, y_train):
    gnb = naive_bayes.GaussianNB()
    gnb_cv_scores = cross_val_score(
        gnb, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    gnb_score = gnb_cv_scores.mean()
    gnb.fit(x_train, y_train)

    return gnb_score, gnb


def neural_classifier(x_train, y_train):
    nn_parameters = {'hidden_layer_sizes': [10, 30, 50], 'activation': ['relu'], 'solver': ['adam'],
                     'learning_rate': ['constant'], 'learning_rate_init': [0.001, 0.01, 0.1, 0.0001], 'max_iter': [200],
                     'alpha': [0.001, 0.0001, 0.01]}
    nn_classifier = neural_network.MLPClassifier()
    grid_nn = RandomizedSearchCV(nn_classifier, nn_parameters,
                                 scoring='accuracy', cv=5, n_jobs=-1, verbose=3)
    grid_nn.fit(x_train, y_train)
    nn_class_final = grid_nn
    params = grid_nn.best_params_
    print('Best Neural nets Parameters:')
    print(params)
    nn_score = grid_nn.best_score_

    return nn_score, nn_class_final


def BDT_classifier(x_train, y_train):
    bdt_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=1),
                                        algorithm="SAMME",
                                        n_estimators=200)
    scores = cross_val_score(bdt_classifier, x_train, y_train, cv=5)
    bdt_score = scores.mean()
    bdt_classifier.fit(x_train, y_train)

    return bdt_score, bdt_classifier

#
# gnb_score, classifier = bayes_classifier(x_train, y_train)
# y_pred = classifier.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('cross val GNB', gnb_score, 'test GNB:', acc)

svm_score, classifier = svm_classifier(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('cross val svm', svm_score, 'test svm:', acc)

# knn_score, classifier = knn_classifier(x_train, y_train)
# y_pred = classifier.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('cross val knn', knn_score,'test knn:', acc)
#
# nn_score, classifier = neural_classifier(x_train, y_train)
# y_pred = classifier.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('cross val nn', nn_score, 'test nn:', acc)
#
# bdt_score, classifier = BDT_classifier(x_train, y_train)
# y_pred = classifier.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('cross val bdt', bdt_score, 'test bdt:', acc)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
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


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=13)
class_names=np.array(['0', '1','2', '3', '4', '5', '6','7','8','9', '10', '11', '12', '13'])
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')


plt.show()