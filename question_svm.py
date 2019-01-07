import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from SVM_classifier import SVM_classification


def get_data():
    file = "all-yquestions-clean.txt"
    data_all = pd.read_csv(file, sep='\t', header=None)
    data_all = shuffle(data_all)
    data = data_all.iloc[:, 2]
    target = data_all.iloc[:, 1]

    mapping = {'socializing': 0, 'factual': 1, 'opinion': 2}
    target = target.map(mapping)

    split = 0.9
    n = int(len(data) * split)
    train_data = data[:n]
    train_target = target[:n]
    test_data = data[n:]
    test_target = target[n:]

    return train_data, train_target, test_data, test_target


# NB for comparison
def NB_classification(train_data, train_target, test_data, test_target):
    text_clf = Pipeline(
        [('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

    text_clf = text_clf.fit(train_data, train_target)

    predicted = text_clf.predict(test_data)
    NB_acc = np.mean(predicted == test_target)
    print("NB accuracy ", NB_acc)

    predicted_tr = text_clf.predict(train_data)
    NB_tr = np.mean(predicted_tr == train_target)
    print("NB accuracy on train ", NB_tr)

    print("NB grid search starting")
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)], 'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3, 1e-4)}

    gs_clf = GridSearchCV(text_clf, parameters)
    gs_clf = gs_clf.fit(train_data, train_target)

    print("NB grid search best score ", gs_clf.best_score_)
    print("NB grid search parameters ", gs_clf.best_params_)

    return NB_acc, gs_clf, NB_tr


if __name__ == "__main__":
    train_data, train_target, test_data, test_target = get_data()
    NB_acc, gs_clf, NB_tr = NB_classification(train_data, train_target, test_data, test_target)
    SVM_acc, gs_clf_svm, SVM_tr = SVM_classification(train_data, train_target, test_data, test_target)

    print("\n\n---Final results---")
    print("NB accuracy ", NB_acc)
    print("NB accuracy on train set ", NB_tr)
    print("NB grid search best score ", gs_clf.best_score_)
    print("NB grid search parameters ", gs_clf.best_params_)
    print("")
    print("SVM accuracy ", SVM_acc)
    print("SVM accuracy on train set ", SVM_tr)
    print("SVM grid search best score ", gs_clf_svm.best_score_)
    print("SVM grid search parameters ", gs_clf_svm.best_params_)
