from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

from nltk import pos_tag, word_tokenize, sent_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import argparse

from sklearn.externals import joblib


class PosTagTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tagged = [pos_tag(word_tokenize(sent)) for sent in X]
        tags_concat = list()
        for sent in tagged:
            tags_concat.append([''.join(el[1]) for el in sent])
        tags = [' '.join(sent) for sent in tags_concat]

        split_tags = [el.split(' ') for el in tags]
        tag_mapping = {'$': 0, "''": 1, '(': 2, ')': 3, ',': 4, '--': 5, '.': 6, ':': 7, 'CC': 8, 'CD': 9, 'DT': 10,
                       'EX': 11, 'FW': 12,
                       'IN': 13, 'JJ': 14, 'JJR': 15, 'JJS': 16, 'LS': 17, 'MD': 18, 'NN': 19, 'NNP': 20, 'NNPS': 21,
                       'NNS': 22,
                       'PDT': 23, 'POS': 24, 'PRP': 25, 'PRP$': 26, 'RB': 27, 'RBR': 28, 'RBS': 29, 'RP': 30, 'SYM': 31,
                       'TO': 32,
                       'UH': 33, 'VB': 34, 'VBD': 35, 'VBG': 36, 'VBN': 37, 'VBP': 38, 'VBZ': 39, 'WDT': 40, 'WP': 41,
                       'WP$': 42, 'WRB': 43, '``': 44, '#': 45}

        for i in range(len(split_tags)):
            for j in range(len(split_tags[i])):
                split_tags[i][j] = tag_mapping.get(split_tags[i][j], split_tags[i][j])

        final_tags = np.zeros([len(split_tags), len(max(split_tags, key=lambda x: len(x)))])
        for i, j in enumerate(split_tags):
            final_tags[i][0:len(j)] = [int(x) for x in j]
        return np.array(final_tags)


class NormTransformer(TransformerMixin):
    def __init__(self, _count):
        self.count = _count

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if X.shape[1] < self.count:
            diff = self.count - X.shape[1]
            add = np.zeros((X.shape[0], diff))
            return sp.hstack([X, add.astype(float)], format='csr')
        elif X.shape[1] > self.count:
            return np.array([lst[:self.count] for lst in X])
        else:
            return X


class AddPosTagTransformer(TransformerMixin):
    def __init__(self, _tags):
        self.tags = _tags

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return sp.hstack([X, self.tags.astype(float)], format='csr')


def get_data(only_questions, data_form, split):
    # file = "all-yquestions-clean.txt"
    file = "data/a/data/sem-train.txt"
    data_all = pd.read_csv(file, sep='\t', header=None)
    data_all = shuffle(data_all)
    data = list(data_all.iloc[:, 2])
    target = data_all.iloc[:, 1]

    if only_questions:
        for i in range(len(data)):
            sents = sent_tokenize(data[i])
            questions = list()
            for s in sents:
                if s.endswith('?'):
                    questions.append(s)
            if len(questions) > 0:
                data[i] = ' '.join(questions)

    tagged = [pos_tag(word_tokenize(sent)) for sent in data]
    tags_concat = list()
    for sent in tagged:
        tags_concat.append([''.join(el[1]) for el in sent])
    tags = [' '.join(sent) for sent in tags_concat]

    if data_form == 'text-pos':
        for i in range(len(data)):
            data[i] = data[i] + ' ' + tags[i]
    elif data_form == 'pos':
        data = tags

    mapping = {'opinion': 0, 'factual': 1, 'socializing': 2}
    target = target.map(mapping)

    n = int(len(data) * split)
    train_data = data[:n]
    train_target = target[:n]
    test_data = data[n:]
    test_target = target[n:]

    return train_data, train_target, test_data, test_target


def get_eval_data(only_questions, data_form):
    file = "data/a/data/sem-dev.txt"
    data_all = pd.read_csv(file, sep='\t', header=None)
    index = data_all.iloc[:, 0]
    data = list(data_all.iloc[:, 1])

    if only_questions:
        for i in range(len(data)):
            sents = sent_tokenize(data[i])
            questions = list()
            for s in sents:
                if s.endswith('?'):
                    questions.append(s)
            if len(questions) > 0:
                data[i] = ' '.join(questions)

    tagged = [pos_tag(word_tokenize(sent)) for sent in data]
    tags_concat = list()
    for sent in tagged:
        tags_concat.append([''.join(el[1]) for el in sent])
    tags = [' '.join(sent) for sent in tags_concat]

    if data_form == 'text-pos':
        for i in range(len(data)):
            data[i] = data[i] + ' ' + tags[i]
    elif data_form == 'pos':
        data = tags

    return index, data


# Naive Bayes baseline
def NB_classification(train_data, train_target, test_data, test_target, stopwords, sc_average):
    if stopwords:
        sw = 'english'
    else:
        sw = None
    text_clf = Pipeline([('vect', CountVectorizer(stop_words=sw, ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer(use_idf=False)), ('clf', MultinomialNB(alpha=0.01))])

    text_clf = text_clf.fit(train_data, train_target)

    predicted = text_clf.predict(test_data)
    print("\nNB accuracy score: ", accuracy_score(test_target, predicted))
    print("NB precision score: ", precision_score(test_target, predicted, average=sc_average))
    print("NB recall: ", recall_score(test_target, predicted, average=sc_average))
    print("NB f1: ", f1_score(test_target, predicted, average=sc_average))

    predicted_tr = text_clf.predict(train_data)
    NB_tr = np.mean(predicted_tr == train_target)
    print("NB accuracy on train ", NB_tr)

    # joblib_file = "question_NB_text-pos.pkl"
    # joblib.dump(text_clf, joblib_file)


def SVM_classification(train_data, train_target, test_data, test_target, stopwords, pipeline, gridsearch, sc_average, eval_ind, eval_data, predict):
    if stopwords:
        sw = 'english'
    else:
        sw = None

    if pipeline == 'standard':
        text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words=sw, ngram_range=(1, 3))), ('tfidf', TfidfTransformer(use_idf=False, norm=None)),
                                 ('clf-svm', SVC(kernel='linear'))])
    elif pipeline == 'standard-pos':
        text_clf_svm = Pipeline([('union', FeatureUnion([("vect", TfidfVectorizer(stop_words=sw, ngram_range=(1, 2), use_idf=False, norm=None)),
                                                         ("pos-features", PosTagTransformer())])),
                                 ('norm', NormTransformer(_count=40000)),
                                 ('clf-svm', SVC(kernel='linear'))])


    text_clf_svm = text_clf_svm.fit(train_data, train_target)

    print("SVM fit completed")

    predicted_svm_tr = text_clf_svm.predict(train_data)
    SVM_tr = np.mean(predicted_svm_tr == train_target)
    print("SVM accuracy on train ", SVM_tr)

    predicted_svm = text_clf_svm.predict(test_data)
    SVM_acc = np.mean(predicted_svm == test_target)
    print("SVM accuracy ", SVM_acc)

    print("\nSVM accuracy score: ", accuracy_score(test_target, predicted_svm))
    print("SVM precision score: ", precision_score(test_target, predicted_svm, average=sc_average))
    print("SVM recall: ", recall_score(test_target, predicted_svm, average=sc_average))
    print("SVM f1: ", f1_score(test_target, predicted_svm, average=sc_average))

    if predict and not gridsearch:
        eval_pred = text_clf_svm.predict(eval_data)
        results = pd.DataFrame(list(zip(eval_ind, eval_pred)))
        results.to_csv('predict_questions0.txt', sep='\t', header=None, index=None)

    # joblib_file = "question_standard-pos_text-pos.pkl"
    # joblib.dump(text_clf_svm, joblib_file)

    data_all = pd.read_csv('sem-dev.txt', header=None, delimiter='\t')
    predicted = text_clf_svm.predict(data_all.iloc[:, 1])
    data_all[2] = pd.Series(predicted)
    mapping = {0: 'opinion', 1: 'factual', 2: 'socializing'}
    data_all[2] = data_all[2].map(mapping)
    data_all.to_csv('results_standard-pos_text-pos.txt', sep='\t', header=False, index=False)

    if gridsearch:
        print("SVM grid search starting")
        if pipeline == 'standard':
            parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                              'vect__stop_words': ['english', None],
                              'tfidf__use_idf': (True, False),
                              'tfidf__norm': ('l1', 'l2', None),
                              'clf-svm__C': (1, 1.5, 2)}
        elif pipeline == 'standard-pos':
            parameters_svm = {'union__vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                              'union__vect__stop_words': ['english', None],
                              'union__vect__use_idf': (True, False),
                              'union__vect__norm': ('l1', 'l2', None),
                              'clf-svm__C': (1, 1.5, 2)}

        gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, verbose=3)
        gs_clf_svm = gs_clf_svm.fit(train_data, train_target)

        print("SVM grid search best score ", gs_clf_svm.best_score_)
        print("SVM grid search parameters ", gs_clf_svm.best_params_)

        predicted_svm_tr = gs_clf_svm.best_estimator_.predict(train_data)
        SVM_tr = np.mean(predicted_svm_tr == train_target)
        print("SVM best param accuracy on train ", SVM_tr)

        predicted_svm = gs_clf_svm.best_estimator_.predict(test_data)
        SVM_acc = np.mean(predicted_svm == test_target)
        print("SVM best param accuracy ", SVM_acc)

        print("\nSVM best param accuracy score: ", accuracy_score(test_target, predicted_svm))
        print("SVM best param precision score: ", precision_score(test_target, predicted_svm, average=sc_average))
        print("SVM best param recall: ", recall_score(test_target, predicted_svm, average=sc_average))
        print("SVM best param f1: ", f1_score(test_target, predicted_svm, average=sc_average))

        if predict:
            eval_pred = gs_clf_svm.best_estimator_.predict(eval_data)
            results = pd.DataFrame(list(zip(eval_ind, eval_pred)))
            results.to_csv('predict_questions0.txt', sep='\t', header=None, index=None)

        # joblib_file = "question_standard_text_questions.pkl"
        # joblib.dump(text_clf_svm, joblib_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-stopwords', default=False)
    parser.add_argument('-only_questions', default=False)  # bool
    parser.add_argument('-data_form', default='text')  # options: text, text-pos, pos
    parser.add_argument('-split', default=0.8)
    parser.add_argument('-NB', default=False)  # bool
    parser.add_argument('-svm_pipeline', default='standard')  # options: standard, standard-pos
    parser.add_argument('-svm_gridsearch', default=False)  # bool
    parser.add_argument('-score_average', default='micro')  # options: micro, macro
    parser.add_argument('-predict', default=False)  # bool, path should be updated in function get_eval_data()
    args = parser.parse_args()

    train_data, train_target, test_data, test_target = get_data(args.only_questions, args.data_form, args.split)
    if args.NB:
        NB_classification(train_data, train_target, test_data, test_target, args.stopwords, args.score_average)

    if args.predict:
        eval_ind, eval_data = get_eval_data(args.only_questions, args.data_form)
        SVM_classification(train_data, train_target, test_data, test_target, args.stopwords, args.svm_pipeline,
                           args.svm_gridsearch, args.score_average, eval_ind, eval_data, args.predict)
    else:
        SVM_classification(train_data, train_target, test_data, test_target, args.stopwords, args.svm_pipeline,
                           args.svm_gridsearch, args.score_average, '', '', args.predict)
