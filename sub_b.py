def clean_data(data):
    import re

    REPLACE_NO_SPACE = re.compile("(\.)|(;)|(:)|(!)|(\')|(\?)|(,)|(\")|(\()|(\))|(\[)|(\])|(=)|(_)|(\*)|(://)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(-)|(/)(&\s\s\s)")

    def preprocess_reviews(text):
        text = [REPLACE_NO_SPACE.sub("", line.lower()) for line in text]
        text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]

        return text

    return preprocess_reviews(data)


class Data:
    def __init__(self):
        self.questions_d = {}
        self.data = None
        self.target = None
        self.tags = None
        self.categories_d = {'True': 0, 'False': 1}

        self.get_data()

    def get_data(self):
        def get_data_from_file(filename, questions_d, categories_d, r=32):
            with open(filename, 'r') as questions_file:
                data = []
                target = []
                tags = []
                lines = questions_file.readlines()

                import random
                random.seed(r)
                random.shuffle(lines)

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

    def split_train_test(self, split=0.5, data_target=None):
        if data_target is None:
            data, target = self.data, self.target
        else:
            data, target = data_target

        n = int(len(data) * split)
        train_data = data[:n]
        train_target = target[:n]

        test_data = data[n:]
        test_target = target[n:]

        return train_data, train_target, test_data, test_target


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


def run_cross_validation(data, target):
    import numpy as np
    from logistic_regression import prep_data
    lr_data = np.array(prep_data(data))

    nb_data = np.array(data)

    positives_accuracy_nb, negatives_accuracy_nb, positives_accuracy_lr, negatives_accuracy_lr = 0., 0., 0., 0.
    negatives_count, positives_count = 0, 0
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(data)

    target = np.array(target)
    nb_preds = []
    lr_preds = []
    for train_index, test_index in loo.split(data):
        train_data, test_data = nb_data[train_index], nb_data[test_index]
        train_target, test_target = target[train_index], target[test_index]
        nb_pred = get_predictions_naive_bayes(train_data, train_target, test_data)

        train_data, test_data = lr_data[train_index], lr_data[test_index]
        lr_pred = get_predictions_logistic_regression(train_data, train_target, test_data)
        if test_target[0] == 0:
            negatives_count += 1.
            if nb_pred[0] == 0:
                negatives_accuracy_nb += 1.
            if lr_pred[0] == 0:
                negatives_accuracy_lr += 1.
        else:
            positives_count += 1.
            if nb_pred[0] == 1:
                positives_accuracy_nb += 1.
            if lr_pred[0] == 1:
                positives_accuracy_lr += 1.
        nb_preds.append(nb_pred[0])
        lr_preds.append(lr_pred[0])

    if positives_count == 0:
        positives_count = 1
    if negatives_count == 0:
        negatives_count = 1

    print('NB p %f (%d/%d) NB n %f (%d/%d) \nLR p %f  (%d/%d) LR n %f (%d/%d) '
          % (positives_accuracy_nb / positives_count, positives_accuracy_nb, positives_count,
             negatives_accuracy_nb / negatives_count, negatives_accuracy_nb, negatives_count,
             positives_accuracy_lr / positives_count, positives_accuracy_lr, positives_count,
             negatives_accuracy_lr / negatives_count, negatives_accuracy_lr, negatives_count))
    return nb_preds, lr_preds, target


def store_preds(tags, preds, title):
    with open('data/b/' + title + '.txt', 'w') as f:
        f.writelines([tag + '\t' + str(pred) + '\n' for tag, pred in zip(tags, preds)])


def main():
    d = Data()
    data, target, tags = d.data, d.target, d.tags

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

    nb_preds, lr_preds, target = run_cross_validation(data, target)
    store_preds(tags, nb_preds, "nb_preds")
    store_preds(tags, lr_preds, "lr_preds")

    # from cosine_similarity import CosineSimilarity
    # cs = CosineSimilarity()
    # cs.d_train()


if __name__ == "__main__":
    main()
