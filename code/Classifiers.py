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

filename = 'training_set.csv'
training_data = Table.read(filename, format='ascii.csv')
print(training_data.info)
print(training_data)

filename = 'training_set_metadata.csv'
meta_training = Table.read(filename, format='ascii.csv')
n_objects = len(meta_training)
target_vec = np.asarray(meta_training['target'])

unique_targets = np.unique(target_vec)
print("There are {} unique targets.".format(len(unique_targets)))
print(unique_targets)

hist_count = np.zeros([len(unique_targets), 1])
for i in range(1, len(unique_targets)):
    hist_count[i, 0] = np.count_nonzero(target_vec[target_vec == unique_targets[i]])

print('Number of objects of each class')
print(hist_count.ravel())

plt.rcdefaults()
objects = ('6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95')
y_pos = np.arange(len(objects))
performance = hist_count.ravel()

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Amount')
plt.xlabel('Object class')
plt.title('Number of objects from each class')
plt.show()

counts = Counter(meta_training['target'])
labels, values = zip(*sorted(counts.items(), key=itemgetter(1)))

#  binarize target_vec  relative to most frequent label
max_label = labels[np.argmax(values)]
bool_labels = target_vec == max_label
bin_labels = bool_labels.astype(int)
features = np.ones(len(bin_labels))

# Train - test split
x_train, x_test, y_train, y_test = train_test_split(features, bin_labels, test_size=0.2, random_state=0)

pca = PCA(n_components='mle', whiten=True, svd_solver="full", random_state=42)
x_train_pca = pca.fit_transform(x_train)


def svm_classifier(x_train, y_train):
    svm_parameters = {'kernel': ['rbf'], 'gamma': [
        0.01, 0.5, 1, 10], 'C': [0.01, 0.5, 1, 10, 100]}
    grid_svm = RandomizedSearchCV(svm.SVC(decision_function_shape='ovr'), svm_parameters,
                                  scoring='accuracy', cv=3, n_jobs=-1, verbose=3)

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
            knn, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
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
        gnb, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
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
    scores = cross_val_score(bdt_classifier, x_train, y_train, cv=3)
    bdt_score = scores.mean()
    bdt_classifier.fit(x_train, y_train)

    return bdt_score, bdt_classifier
