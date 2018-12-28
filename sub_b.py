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

                last_question = ''
                answers = []
                for line in lines:
                    prts = line.split('\t')
                    tag = prts[0]
                    cat = prts[1]
                    text = prts[2].strip()
                    if cat == 'NA':
                        if last_question != '':
                            questions_d[tag] = answers
                        last_question = text
                        answers = []
                    if cat == 'True' or cat == 'False':
                        answers.append(tag)
                        data.append(text)
                        target.append(categories_d[cat])
                        tags.append(tag)
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
        with open('data/b/b-factual-q-web-results-train.txt', 'r') as f_train, open(
                'data/b/b-factual-q-web-results-test.txt', 'r') as f_test:
            snippets = {}
            lines = f_train.readlines() + f_test.readlines()
            for line in lines:
                prts = line.split('\t')
                tag = prts[0].strip()
                snippet = prts[1].strip()
                snippets[tag] = snippet
        return snippets


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


def get_cross_validation_predictions(data, target, method):
    import numpy as np
    data = np.array(data)
    target = np.array(target)

    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(data)

    preds = []
    for train_index, test_index in loo.split(data):
        train_target, test_target = target[train_index], target[test_index]
        train_data, test_data = data[train_index], data[test_index]
        pred = method(train_data, train_target, test_data)

        preds.append(pred[0])

    return preds, target


def get_cross_validation_predictions_lr(data, target):
    return get_cross_validation_predictions(data, target, get_predictions_logistic_regression)


def get_cross_validation_predictions_nb(data, target):
    return get_cross_validation_predictions(data, target, get_predictions_naive_bayes)


def get_cross_validation_predictions_cs(data, target):
    return get_cross_validation_predictions(data, target, get_predictions_cosine_similarity)


def store_preds(tags, preds, title):
    with open('data/b/' + title + '.txt', 'w') as f:
        f.writelines([tag + '\t' + str(pred) + '\n' for tag, pred in zip(tags, preds)])


def main():
    d = Data()
    data, target, tags = d.data, d.target, d.tags

    from logistic_regression import prep_data
    lr_data = prep_data(data)

    ## get_accuracy_naive_bayes(data, target)
    # Accuracy %d 0.6557377049180327

    ## get_accuracy_logistic_regression(data, target):
    # from logistic_regression import prep_data
    # features_nd = prep_data(data)
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(features_nd, target, train_size=0.5, test_size=0.3,
    #                                                     random_state=1234, shuffle=False)
    # # print(y_test[0], X_test[0])
    # acc=get_accuracy_logistic_regression(X_train, y_train, X_test, y_test)
    # print(acc)
    # [Accuracy]: 0.6739130434782609

    # nb_preds, lr_preds, target = run_cross_validation(data, target)
    # store_preds(tags, nb_preds, "nb_preds")
    # store_preds(tags, lr_preds, "lr_preds")

    from cosine_similarity import CosineSimilarity
    cs = CosineSimilarity()
    similarity_scores, tags = cs.predict(d)
    store_preds(tags, similarity_scores, "cs_scores")

    import numpy as np
    data = np.array([[s] for s in similarity_scores])

    target = np.array(target)
    cs_preds, target = get_cross_validation_predictions_cs(data, target)
    store_preds(tags, cs_preds, "cs_preds")


if __name__ == "__main__":
    main()
