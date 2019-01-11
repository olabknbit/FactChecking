import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def SVM_classification(train_data, train_target, test_data, test_target):
    text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                             ('clf-svm',
                              SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

    text_clf_svm = text_clf_svm.fit(train_data, train_target)

    predicted_svm = text_clf_svm.predict(test_data)
    SVM_acc = np.mean(predicted_svm == test_target)
    print("SVM accuracy ", SVM_acc)

    predicted_svm_tr = text_clf_svm.predict(train_data)
    SVM_tr = np.mean(predicted_svm_tr == train_target)
    print("SVM accuracy on train ", SVM_tr)

    print("SVM grid search starting")
    parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)], 'tfidf__use_idf': (True, False),
                      'clf-svm__alpha': (1e-2, 1e-3, 1e-4)}

    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm)
    gs_clf_svm = gs_clf_svm.fit(train_data, train_target)

    print("SVM grid search best score ", gs_clf_svm.best_score_)
    print("SVM grid search parameters ", gs_clf_svm.best_params_)

    return SVM_acc, gs_clf_svm, SVM_tr


class SVM_classifier:
    def __init__(self, with_pipeline=False):
        if with_pipeline:
            self.classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                                        ('clf-svm',
                                         SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5,
                                                       random_state=42))])
        else:
            self.classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)

    def train(self, train_data, train_target):
        self.classifier.fit(train_data, train_target)
        return self.classifier

    def get_predictions(self, test_data):
        if self.classifier is None:
            print("Error - please train first")
            exit(1)
        predictions = self.classifier.predict(test_data)
        return predictions

    def run_svm_accuracy(self, test_data, test_target):
        predictions = self.get_predictions(test_data)
        from sklearn.metrics import accuracy_score
        return accuracy_score(test_target, predictions)
