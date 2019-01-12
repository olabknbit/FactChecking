from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


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
