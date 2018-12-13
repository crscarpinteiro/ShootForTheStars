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


def svm_classifier(x_train, y_train):
    svm_parameters = [{'kernel': ['rbf'], 'gamma': [
        0.01, 0.5, 1, 10], 'C': [0.01, 0.5, 1, 10, 100]}]
    grid_svm = GridSearchCV(svm.SVC(decision_function_shape='ovr'), svm_parameters,
                            probability=True, scoring='accuracy', cv=5, n_jobs=-1, verbose=3)
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

    return gnb_score, gnb


def neural_classifier(x_train, y_train):
    nn_parameters = {'hidden_layer_sizes': [10, 30, 50], 'activation': ['relu'], 'solver': ['adam'],
                     'learning_rate': ['constant'], 'learning_rate_init': [0.001, 0.01, 0.1, 0.0001], 'max_iter': [200],
                     'alpha': [0.001, 0.0001, 0.01]}
    nn_class = neural_network.MLPClassifier()
    grid_nn = GridSearchCV(nn_class, nn_parameters,
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=3)
    grid_nn.fit(x_train, y_train)
    nn_class_final = grid_nn
    params = grid_nn.best_params_
    print('Best Neural nets Parameters:')
    print(params)
    nn_score = grid_nn.best_score_

    return nn_score, nn_class_final


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


def print_error_rate(err):
    print('Error rate: Training: %.4f - Test: %.4f' % err)


def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x == 1 else -1 for x in miss]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(list(range(0, 450, 50)))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')


clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
