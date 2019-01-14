# dirs and filenames
PREFIX = 'data/b/'
PREDS_DIR = 'preds/'
FEATURES_DIR = 'features/'
RESULTS_DIR = 'results/'
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
ADD_ON = ''

# methods
LR = 'lr'
NB = 'nb-gaussian'
CS = 'cs-with-svm'
SVM = 'svm-no-stop-words-removal'

# features
WEB = 'web'
COUNT_VECTORIZER = 'count_vectorizer'


def get_data_filename(mode):
    return PREFIX + DATA_DIR + 'b-factual-q-a-clean-' + mode + '.txt'


def get_data_train_filename():
    return get_data_filename('train')


def get_data_test_filename():
    return get_data_filename('test')


def get_pred_filename(method):
    return PREFIX + PREDS_DIR + method + '_preds' + ADD_ON + '.txt'


def get_features_filename(which):
    return PREFIX + FEATURES_DIR + which + '_features.txt'


def get_serial_filename(method, q_tag=None):
    if q_tag is None:
        q_tag = ""
    else:
        q_tag = "_" + q_tag
    return PREFIX + MODELS_DIR + method + q_tag + ".0"


def get_cs_pred_filename():
    return get_pred_filename(CS)


def get_nb_pred_filename():
    return get_pred_filename(NB)


def get_lr_pred_filename():
    return get_pred_filename(LR)


def get_svm_pred_filename():
    return get_pred_filename(SVM)


def get_web_features_filename():
    return get_features_filename(WEB)


def get_serial_filename_nb(q_tag=None):
    return get_serial_filename(NB, q_tag=q_tag)


def get_serial_filename_svm(q_tag=None):
    return get_serial_filename(SVM, q_tag=q_tag)


def get_serial_filename_lr(q_tag=None):
    return get_serial_filename(LR, q_tag=q_tag)


def get_serial_filename_cs(q_tag=None):
    return get_serial_filename(CS, q_tag=q_tag)


def get_results_filename(methods):
    return PREFIX + RESULTS_DIR + '-'.join(methods) + ADD_ON + '.txt'


def split_train_test(data, target, split=0.5):
    n = int(len(data) * split)
    train_data = data[:n]
    train_target = target[:n]

    test_data = data[n:]
    test_target = target[n:]

    return train_data, train_target, test_data, test_target


def get_question_tag(tag):
    prts = tag.split('_')
    Q = prts[0]
    R = prts[1]
    return Q + '_' + R


class Data:
    def __init__(self):
        self.questions_d = {}
        self.data = None
        self.target = None
        self.tags = None
        self.categories_d = {'True': 0, 'False': 1}

        self.get_data()

    def get_data(self):
        def get_data_from_file(filename, questions_d, categories_d):
            with open(filename, 'r') as questions_file:
                data = []
                target = []
                tags = []
                lines = questions_file.readlines()

                import random
                random.seed(32)
                random.shuffle(lines)

                last_question_tag = ''
                answers = []
                for line in lines:
                    prts = line.split('\t')
                    tag = prts[0]
                    cat = prts[1]
                    text = prts[2].strip()
                    if cat == 'NA':
                        if last_question_tag != '':
                            questions_d[last_question_tag] = answers
                        last_question_tag = tag
                        answers = []
                    if cat == 'True' or cat == 'False':
                        answers.append(tag)
                        data.append(text)
                        target.append(categories_d[cat])
                        tags.append(tag)
                questions_d[last_question_tag] = answers
            return data, target, tags

        train_data, train_target, train_tags = get_data_from_file(get_data_train_filename(), self.questions_d,
                                                                  self.categories_d)
        test_data, test_target, test_tags = get_data_from_file(get_data_test_filename(), self.questions_d,
                                                               self.categories_d)

        self.data = train_data + test_data
        self.target = train_target + test_target
        self.tags = train_tags + test_tags

    def split_train_test(self, split=0.5):
        data, target = self.data, self.target
        return split_train_test(data, target, split)

    def get_q_query_snippet_answer_pairs(self):
        question_snippets = self.get_question_snippets()
        sorted_question_snippets = []
        for tag in self.tags:
            q_tag = get_question_tag(tag)
            sorted_question_snippets.append(question_snippets[q_tag])
        return sorted_question_snippets, self.data, self.tags

    def get_question_snippets(self):
        with open('data/b/b-factual-q-web-results-train.txt', 'r') as f_train, \
                open('data/b/b-factual-q-web-results-test.txt', 'r') as f_test:
            snippets = {}
            lines = f_train.readlines() + f_test.readlines()
            for line in lines:
                prts = line.split('\t')
                tag = prts[0].strip()
                snippet = prts[1].strip()
                snippets[tag] = snippet
        return snippets

    def is_question(self, tag):
        return tag in self.questions_d

    def get_all_aswers_to_question(self, tag):
        return self.questions_d[tag]


def read_features(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        tags = []
        features = []
        for line in lines:
            prts = line.split('\t')
            tag = prts[0]
            feature = [float(f.strip()) for f in prts[1:]]
            features.append(feature)
            tags.append(tag)
    return features, tags


def read_preds(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        tags = []
        preds = []
        for line in lines:
            prts = line.split('\t')
            tag = prts[0]
            pred = int(prts[1].strip())
            preds.append(pred)
            tags.append(tag)
    return preds, tags


def get_predictions_naive_bayes(train_data, train_target, test_data, q_tag=None):
    from naive_bayes import NaiveBayes
    nb = NaiveBayes(serial_filename=get_serial_filename_nb(q_tag=q_tag))
    nb.train(train_data, train_target)
    return nb.get_predictions(test_data)


def get_accuracy_naive_bayes(train_data, train_target, test_data, test_target):
    predictions = get_predictions_naive_bayes(train_data, train_target, test_data)
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_target, predictions)


def get_predictions_logistic_regression(train_data, train_target, test_data, q_tag=None):
    from logistic_regression import LogisticRegression
    lr = LogisticRegression(serial_filename=get_serial_filename_lr(q_tag=q_tag))
    lr.train(train_data, train_target)
    return lr.get_predictions(test_data)


def get_accuracy_logistic_regression(train_data, train_target, test_data, test_target):
    predictions = get_predictions_logistic_regression(train_data, train_target, test_data)
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_target, predictions)


def _get_predictions_svm_helper(train_data, train_target, test_data, with_pipeline=True, serial_filename=None):
    from SVM_classifier import SVM_classifier
    svm = SVM_classifier(with_pipeline, serial_filename=serial_filename)
    svm.train(train_data, train_target)
    return svm.get_predictions(test_data)


def get_predictions_cosine_similarity(train_data, train_target, test_data, q_tag=None):
    return _get_predictions_svm_helper(train_data, train_target, test_data, with_pipeline=False,
                                       serial_filename=get_serial_filename_cs(q_tag=q_tag))


def get_predictions_svm(train_data, train_target, test_data, q_tag=None):
    return _get_predictions_svm_helper(train_data, train_target, test_data, with_pipeline=True,
                                       serial_filename=get_serial_filename_svm(q_tag=q_tag))


def get_accuracy_svm(train_data, train_target, test_data, test_target):
    predictions = get_predictions_svm(train_data, train_target, test_data)
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_target, predictions)


def get_all_questions_belonging_to_thread(data_obj, tags, index):
    tag = tags[index]
    if not data_obj.is_question(tag):
        q_tag = get_question_tag(tag)
    else:
        q_tag = tag
    tlo = data_obj.get_all_aswers_to_question(q_tag)
    indexes = [tags.index(t) for t in tlo if tag != t]
    indexes = [i if index > i else i - 1 for i in indexes]
    return indexes, q_tag


def get_cross_validation_predictions(data_obj, data, target, tags, method):
    import numpy as np
    data = np.array(data)
    target = np.array(target)

    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(data)

    preds = []
    for train_index, test_index in loo.split(data):
        indexes_to_leave_out, q_tag = get_all_questions_belonging_to_thread(data_obj, tags, index=list(test_index)[0])
        train_index = np.delete(train_index, indexes_to_leave_out, 0)
        train_target, test_target = target[train_index], target[test_index]
        train_data, test_data = data[train_index], data[test_index]
        pred = method(train_data, train_target, test_data, q_tag)
        preds.append(pred[0])

    return preds, target


def get_cross_validation_predictions_lr(data_obj, data, target, tags):
    return get_cross_validation_predictions(data_obj, data, target, tags, get_predictions_logistic_regression)


def get_cross_validation_predictions_nb(data_obj, data, target, tags):
    return get_cross_validation_predictions(data_obj, data, target, tags, get_predictions_naive_bayes)


def get_cross_validation_predictions_cs(data_obj, data, target, tags):
    return get_cross_validation_predictions(data_obj, data, target, tags, get_predictions_cosine_similarity)


def get_cross_validation_predictions_svm(data_obj, data, target, tags):
    return get_cross_validation_predictions(data_obj, data, target, tags, get_predictions_svm)


def store_preds(tags, preds, title):
    with open(title, 'w') as f:
        f.writelines([tag + '\t' + str(pred) + '\n' for tag, pred in zip(tags, preds)])


def get_lr_preds(data_obj):
    import os
    preds_filename = get_lr_pred_filename()
    exists = os.path.isfile(preds_filename)
    if exists:
        lr_preds, tags = read_preds(preds_filename)
    else:
        from logistic_regression import prep_data
        lr_features = prep_data(data_obj.data)
        tags = data_obj.tags
        lr_preds, target = get_cross_validation_predictions_lr(data_obj, lr_features, data_obj.target, tags)
        store_preds(data_obj.tags, lr_preds, preds_filename)
    return lr_preds, tags


def get_svm_preds(data_obj):
    import os
    filename = get_svm_pred_filename()
    exists = os.path.isfile(filename)
    if exists:
        preds, tags = read_preds(filename)
    else:
        tags = data_obj.tags
        preds, target = get_cross_validation_predictions_svm(data_obj, data_obj.data, data_obj.target, tags)
        store_preds(data_obj.tags, preds, filename)
    return preds, tags


def get_nb_preds(data_obj):
    import os
    filename = get_nb_pred_filename()
    exists = os.path.isfile(filename)
    if exists:
        nb_preds, tags = read_preds(filename)
    else:
        from naive_bayes import prep_data
        features = prep_data(data_obj.data)
        nb_preds, target = get_cross_validation_predictions_nb(data_obj, features, data_obj.target, data_obj.tags)
        tags = data_obj.tags
        store_preds(data_obj.tags, nb_preds, filename)
    return nb_preds, tags


def get_cs_preds(data_obj):
    import os
    filename = get_cs_pred_filename()
    exists = os.path.isfile(filename)
    if exists:
        cs_preds, tags = read_preds(filename)
    else:
        similarity_scores, tags = read_features(get_web_features_filename())
        import numpy as np
        data = np.array(similarity_scores)
        target = np.array(data_obj.target)
        cs_preds, target = get_cross_validation_predictions_cs(data_obj, data, target, tags)
        store_preds(tags, cs_preds, filename)
    return cs_preds, tags


def sort_tags(preds, ts, tags):
    new_preds = []
    for tag in tags:
        new_preds.append(preds[ts.index(tag)])
    return new_preds, tags


def multifaceted_predictions(data_obj, methods):
    def get_baseline(coefs):
        return sum(coefs) / 2

    def get_preds(various_preds, coefs):
        baseline = get_baseline(coefs)
        preds = []
        for var_p in various_preds:
            s = sum([p * c for p, c in zip(var_p, coefs)])
            if s > baseline:
                pred = 1
            else:
                pred = 0
            preds.append(pred)
        return preds

    various_preds = []
    coefs = []
    if LR in methods:
        coef = 1
        lr_preds, lr_tags = sort_tags(*get_lr_preds(data_obj), data_obj.tags)
        various_preds.append(lr_preds)
        coefs.append(coef)
    if NB in methods:
        coef = 1
        nb_preds, nb_tags = sort_tags(*get_nb_preds(data_obj), data_obj.tags)
        various_preds.append(nb_preds)
        coefs.append(coef)
    if CS in methods:
        coef = 1
        cs_preds, cs_tags = sort_tags(*get_cs_preds(data_obj), data_obj.tags)
        various_preds.append(cs_preds)
        coefs.append(coef)
    if SVM in methods:
        coef = 1
        svm_preds, svm_tags = sort_tags(*get_svm_preds(data_obj), data_obj.tags)
        various_preds.append(svm_preds)
        coefs.append(coef)

    return get_preds(zip(*various_preds), coefs)


def multifaceted_accuracy(data_obj, methods):
    predictions = multifaceted_predictions(data_obj, methods)

    from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, \
        jaccard_similarity_score
    target = data_obj.target
    # swap 0s and 1s to get positive class be True instead of False
    target = [0 if i == 1 else 1 for i in target]
    predictions = [0 if i == 1 else 1 for i in predictions]
    return accuracy_score(target, predictions), \
           precision_score(target, predictions), \
           average_precision_score(target, predictions), \
           recall_score(target, predictions), \
           jaccard_similarity_score(target, predictions)


def save_results_to_file(results, methods):
    with open(get_results_filename(methods), 'w') as f:
        f.write(results)


def main():
    d = Data()
    methods = [NB, SVM, CS]
    methods.sort()
    accuracy, precision, AP, recall, IoU = multifaceted_accuracy(d, methods)
    metrics = "accuracy (A): %f\nprecision (P): %f\naverage precision (AP): %f\nrecall (R): %f\njaccard (IoU): %f" \
              % (accuracy, precision, AP, recall, IoU)
    print(metrics)
    save_results_to_file(metrics, methods)


if __name__ == "__main__":
    main()
