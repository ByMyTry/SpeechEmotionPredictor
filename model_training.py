import warnings
import pickle
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn import preprocessing
from sklearn.externals import joblib
from plotly.offline import plot
from plotly.graph_objs import Scatter

from data_extraction import get_speech_files_info, extract_from_wavs_infos, extract_from_wav
from config import emotions_map

LOG_INFO_TAG = '\n[INFO]: '

RANDOM_STATE = 1
NEIGHBORS_LOWER_BOUND = 2
NEIGHBORS_UPPER_BOUND = 10#50


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        X, y = load_data('data.txt')
        X, y = make_data_preproces(X, y)
        best_k = calc_best_k(X, y)
        X_train, y_train, X_test, y_test = separate_tt(X, y, 0.8)
        model = fit_model(X_train, y_train, best_k)
        test_on_data(X_test, y_test, model)
        test_on_file(model, 'test.wav')
        save_model(model, 'model.pkl')


def load_data(data_file_name):
    # print LOG_INFO_TAG + 'data extraction'
    # wav_files_info = get_speech_files_info()
    # Xy = extract(wav_files_info)
    # np.savetxt('data.txt', Xy)
    Xy = np.loadtxt(data_file_name, dtype=np.float64)

    X, y = Xy[:, :-1], Xy[:, -1]
    print X.shape
    return X, y


def make_data_preproces(X, y):
    print LOG_INFO_TAG + 'data scaling'
    # X = preprocessing.normalize(X)
    X = preprocessing.scale(X)
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    return X, y


def calc_best_k(X, y):
    print LOG_INFO_TAG + 'n_neighbors choice & cross validation'
    best_k, max_score, max_scores = None, None, None
    scs, ks = [], []
    for _k in xrange(NEIGHBORS_LOWER_BOUND, NEIGHBORS_UPPER_BOUND):
        cv = KFold(shuffle=True, n_splits=5, random_state=RANDOM_STATE)
        neigh = KNeighborsClassifier(n_neighbors=_k)
        scores = cross_val_score(neigh, X=X, y=y, cv=cv)
        score = scores.mean()
        if score > max_score:
            print score, _k
            max_score = score
            max_scores = scores
            best_k = _k
        scs.append(score), ks.append(_k)
    print LOG_INFO_TAG + 'best params:', max_score, best_k, max_scores
    trace = Scatter(
        x=ks,
        y=scs,
        mode='lines',
        name='scores(neighbors)'
    )
    plot([trace])
    return best_k


def separate_tt(X, y, coef):
    sep_index = int(len(X) * coef)
    print X.shape, len(X), sep_index
    X_train, y_train, X_test, y_test = X[:sep_index], y[:sep_index], X[sep_index:], y[sep_index:]
    return X_train, y_train, X_test, y_test


def fit_model(X_train, y_train, best_k):
    print LOG_INFO_TAG + 'model training'

    model = KNeighborsClassifier(n_neighbors=best_k)  # LogisticRegression()# GaussianNB()#
    model.fit(X_train, y_train)
    print LOG_INFO_TAG + 'model training success'
    return model


def test_on_data(X_test, y_test, model):
    print LOG_INFO_TAG + 'model tests'
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)
    # train_predicted = model.predict(X_train)

    # for i in xrange(100):
    #     print expected[0+i], model.predict_proba(X[0+i])

    # summarize the fit of the model
    print metrics.classification_report(expected, predicted)
    # print metrics.classification_report(y_train, train_predicted)
    # print(metrics.confusion_matrix(expected, predicted))


def test_on_file(model, test_file_name):
    print LOG_INFO_TAG + test_file_name+' predict'
    x0 = np.array(extract_from_wav(test_file_name, 1200))
    x0 = preprocessing.scale(x0)
    # print x0
    print model.predict_proba(x0), model.predict(x0), emotions_map[int(model.predict(x0))]

    print model.kneighbors(x0, 4)


def save_model(model, model_file_name):
    print LOG_INFO_TAG + 'model serialization to "'+model_file_name+'" file'
    # pickle.dump(model, 'model.pkl')
    joblib.dump(model, model_file_name)

if __name__ == '__main__':
    main()
