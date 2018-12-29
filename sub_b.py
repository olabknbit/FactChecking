CS_PREDS_FILENAME = 'data/b/cs_preds.txt'
LR_PREDS_FILENAME = 'data/b/lr_preds.txt'
NB_PREDS_FILENAME = 'data/b/nb_preds.txt'
CS_SCORES_FILENAME = 'data/b/cs_scores.txt'


def clean_data(data):
    import re

    REPLACE_NO_SPACE = re.compile("(\.)|(;)|(:)|(!)|(\')|(\?)|(,)|(\")|(\()|(\))|(\[)|(\])|(=)|(_)|(\*)|(://)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(-)|(/)(&\s\s\s)")

    def preprocess_reviews(text):
        text = [REPLACE_NO_SPACE.sub("", line.lower()) for line in text]
        text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]

        return text

    return preprocess_reviews(data)


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

        train_data, train_target, train_tags = get_data_from_file('data/b/b-factual-q-a-clean-test.txt',
                                                                  self.questions_d,
                                                                  self.categories_d)
        test_data, test_target, test_tags = get_data_from_file('data/b/b-factual-q-a-clean-train.txt', self.questions_d,
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


def get_predictions_naive_bayes(train_data, train_target, test_data):
    from naive_bayes import NaiveBayes
    nb = NaiveBayes()
    nb.train(train_data, train_target)
    return nb.get_predictions(test_data)


def get_accuracy_naive_bayes(train_data, train_target, test_data, test_target):
    predictions = get_predictions_naive_bayes(train_data, train_target, test_data)
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_target, predictions)


def get_predictions_logistic_regression(train_data, train_target, test_data):
    from logistic_regression import LogisticRegression
    lr = LogisticRegression()
    lr.train(train_data, train_target)
    return lr.get_predictions(test_data)


def get_accuracy_logistic_regression(train_data, train_target, test_data, test_target):
    predictions = get_predictions_logistic_regression(train_data, train_target, test_data)
    from sklearn.metrics import accuracy_score
    return accuracy_score(test_target, predictions)


def get_predictions_cosine_similarity(train_data, train_target, test_data):
    return get_predictions_logistic_regression(train_data, train_target, test_data)


def get_all_questions_belonging_to_thread(data_obj, tags, index):
    tag = tags[index]
    if not data_obj.is_question(tag):
        q_tag = get_question_tag(tag)
    else:
        q_tag = tag
    tlo = data_obj.get_all_aswers_to_question(q_tag)
    indexes = [tags.index(t) for t in tlo if tag != t]
    indexes = [i if index > i else i - 1 for i in indexes]
    return indexes


def get_cross_validation_predictions(data_obj, data, target, tags, method):
    import numpy as np
    data = np.array(data)
    target = np.array(target)

    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(data)

    preds = []
    for train_index, test_index in loo.split(data):
        indexes_to_leave_out = get_all_questions_belonging_to_thread(data_obj, tags, index=list(test_index)[0])
        train_index = np.delete(train_index, indexes_to_leave_out, 0)
        train_target, test_target = target[train_index], target[test_index]
        train_data, test_data = data[train_index], data[test_index]
        pred = method(train_data, train_target, test_data)

        preds.append(pred[0])

    return preds, target


def get_cross_validation_predictions_lr(data_obj, data, target, tags):
    return get_cross_validation_predictions(data_obj, data, target, tags, get_predictions_logistic_regression)


def get_cross_validation_predictions_nb(data_obj, data, target, tags):
    return get_cross_validation_predictions(data_obj, data, target, tags, get_predictions_naive_bayes)


def get_cross_validation_predictions_cs(data_obj, data, target, tags):
    return get_cross_validation_predictions(data_obj, data, target, tags, get_predictions_cosine_similarity)


def store_preds(tags, preds, title):
    with open(title, 'w') as f:
        f.writelines([tag + '\t' + str(pred) + '\n' for tag, pred in zip(tags, preds)])


def get_lr_preds(data_obj):
    import os
    exists = os.path.isfile(LR_PREDS_FILENAME)
    if exists:
        lr_preds, tags = read_preds(LR_PREDS_FILENAME)

    else:
        from logistic_regression import prep_data
        lr_data = prep_data(data_obj.data)
        tags = data_obj.tags
        lr_preds, target = get_cross_validation_predictions_lr(data_obj, lr_data, data_obj.target, tags)
        store_preds(data_obj.tags, lr_preds, LR_PREDS_FILENAME)
    return lr_preds, tags


def get_nb_preds(data_obj):
    import os
    exists = os.path.isfile(NB_PREDS_FILENAME)
    if exists:
        nb_preds, tags = read_preds(NB_PREDS_FILENAME)
    else:
        nb_preds, target = get_cross_validation_predictions_nb(data_obj, data_obj.data, data_obj.target, data_obj.tags)
        tags = data_obj.tags
        store_preds(data_obj.tags, nb_preds, NB_PREDS_FILENAME)
    return nb_preds, tags


def get_cs_preds(data_obj):
    import os
    exists = os.path.isfile(CS_PREDS_FILENAME)
    CS_SCORES_FILENAME = 'data/b/web_features.txt'
    if not exists:
        cs_preds, tags = read_preds(CS_PREDS_FILENAME)
        return cs_preds, tags
    else:
        similarity_scores, tags = read_features(CS_SCORES_FILENAME)
    #     TODO: fix web scrapping before using
    # else:
    #     from cosine_similarity import CosineSimilarity
    #     cs = CosineSimilarity()
    #     similarity_scores, tags = cs.predict(data_obj)
    #     store_preds(tags, similarity_scores, CS_SCORES_FILENAME)

    import numpy as np
    data = np.array(similarity_scores)
    target = np.array(data_obj.target)
    cs_preds, target = get_cross_validation_predictions_cs(data_obj, data, target, tags)
    store_preds(tags, cs_preds, 'data/b/web_features_pred.txt')
    return cs_preds, tags


def sort_tags(preds, ts, tags):
    new_preds = []
    for tag in tags:
        new_preds.append(preds[ts.index(tag)])
    return new_preds, tags


def multifaceted_predictions(data_obj):
    lr_preds, lr_tags = sort_tags(*get_lr_preds(data_obj), data_obj.tags)
    nb_preds, nb_tags = sort_tags(*get_nb_preds(data_obj), data_obj.tags)
    cs_preds, cs_tags = sort_tags(*get_cs_preds(data_obj), data_obj.tags)

    A = 1
    B = 1
    C = 1

    preds = []
    for lr_pred, nb_pred, cs_pred in zip(lr_preds, nb_preds, cs_preds):
        if lr_pred * A + nb_pred * B + cs_pred * C > (A + B + C) / 3:
            pred = 1
        else:
            pred = 0
        preds.append(pred)
    return preds


def multifaceted_accuracy(data_obj):
    predictions = multifaceted_predictions(data_obj)
    from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, \
        jaccard_similarity_score
    return accuracy_score(data_obj.target, predictions), \
           precision_score(data_obj.target, predictions), \
           average_precision_score(data_obj.target, predictions), \
           recall_score(data_obj.target, predictions), \
           jaccard_similarity_score(data_obj.target, predictions)


def main():
    d = Data()
    accuracy, precision, AP, recall, IoU = multifaceted_accuracy(d)
    print("accuracy (A): %f\nprecision (P): %f\naverage precision (AP): %f\nrecall (R): %f\njaccard (IoU): %f"
          % (accuracy, precision, AP, recall, IoU))


if __name__ == "__main__":
    main()
