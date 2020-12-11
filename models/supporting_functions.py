import numpy as np
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import nltk

# download supporting files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def average_accuracy(Y_real, Y_predictions):
    """
    Computes average accuracy across all categories:

    inputs:
    Y_real: correct classifications
    Y_predictions: predicted classifications

    Return:
    average accuracy
    """

    return (Y_real == Y_predictions).sum().sum() / Y_real.size


def average_precision(Y_real, Y_predictions, avg='macro'):
    """
    Computes average precision across all categories:

    inputs:
    Y_real: correct classifications
    Y_predictions: predicted classifications
    avg: Default == 'macro'. See f1_score doc for other options

    Return:
    accuracy
    """

    prec = []

    # iterate
    for i, category in enumerate(Y_real):
        prec.append(precision_score(Y_real[category], Y_predictions[:, i], average=avg))

    return np.mean(prec)


def average_recall(Y_real, Y_predictions, avg='macro'):
    """
    Computes average recall across all categories:

    inputs:
    Y_real: correct classifications
    Y_predictions: predicted classifications
    avg: Default == 'macro'. See f1_score doc for other options

    Return:
    accuracy
    """

    rec = []

    # iterate
    for i, category in enumerate(Y_real):
        rec.append(recall_score(Y_real[category], Y_predictions[:, i], average=avg))

    return np.mean(rec)


def average_f1(Y_real, Y_predictions, avg='macro'):
    """
    Computes average f1-score across all categories:

    inputs:
    Y_real: correct classifications
    Y_predictions: predicted classifications
    avg: Default == 'macro'. See f1_score doc for other options

    Return:
    accuracy
    """

    f1 = []

    # iterate
    for i, category in enumerate(Y_real):
        f1.append(f1_score(Y_real[category], Y_predictions[:, i], average=avg))

    return np.mean(f1)


# noinspection SpellCheckingInspection
def naive_predictions(Y_test):
    """
    Creates naive predictions, by predicting the majority class

    inputs:
    Y_test: (correct) targets

    return
    naive_pred: naive predictions

    """

    # initialize prediction:
    naive_pred = Y_test.copy()

    # update, column by column
    for category in Y_test.columns:

        # check accuracy of predicting TRUE to all
        categ_acc = Y_test[category].mean()

        # predict the majority class:
        if categ_acc > 0.5:
            naive_pred[category] = 1
        else:
            naive_pred[category] = 0

    return naive_pred


def print_classification_report(Y_real, Y_predictions):
    """
    Prints classification report per category:

    inputs:
    Y_real: correct classifications
    Y_predictions: predicted classifications

    Return:
    none
    """
    # create lists:
    acc = []
    f1 = []

    # iterate
    for i, category in enumerate(Y_real):
        print("Category: ", category)
        print(classification_report(Y_real[category], Y_predictions[:, i]))
        f1.append(f1_score(Y_real[category], Y_predictions[:, i], average='macro'))
        acc.append(accuracy_score(Y_real[category], Y_predictions[:, i]))

    print("-------------------------------------------------------")
    print("Mean accuracy score: {:.4f}".format(np.mean(acc)))
    print("Mean f1-score (macro): {:.4f}".format(np.mean(f1)))


def tokenize(text, stop_words=stopwords.words("english")):
    """
        Converts a text to tokens, by the following pipeline:

        * Normalize case and remove punctuations
        * split into words
        * remove stop words (english)
        * lemmatize
        * stems


        input:
        text: a string
        stop_words: list of stop words. Default = stopwords.word("english") from nltk

        :return:
        tokenize string, in a list

        """

    # prep nltk transformation objects

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmed = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word not in stop_words]

    # Reduce words to their stems
    stemmed = [stemmer.stem(word) for word in lemmed]

    return stemmed
