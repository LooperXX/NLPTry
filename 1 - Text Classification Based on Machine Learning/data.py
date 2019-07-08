import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopwords_english = stopwords.words('english')


# 考虑用频度阈值和停用词表进行过滤 http://www.cnblogs.com/amiza/p/10407801.html

class SAOMR:
    def __init__(self, path='data/train.tsv', classes=5, shuffle=True):
        self.path = path
        self.classes = classes
        self.shuffle = shuffle
        self.pre_process()

    def shuffle_data(self):
        data = np.array([self.X_data, self.Y_data]).transpose(1, 0)
        np.random.shuffle(data)
        data = data.transpose(1, 0)
        self.X_data = data[0]
        self.Y_data = data[1]

    def data_split(self, radio=None):
        if radio is None:
            radio = [0.8, 0.2]
        data_size = self.X_data.shape[0]
        train_size = int(data_size * radio[0])
        # validate_size = int(data_size * radio[1])
        self.X_train = self.X_data[:train_size]
        self.Y_train = self.Y_one_hot[:, :train_size]
        self.X_validate = self.X_data[train_size:]
        self.Y_validate = self.Y_data[train_size:]
        # self.X_validate = self.X_data[train_size:train_size + validate_size]
        # self.Y_validate = self.Y_data[train_size:train_size + validate_size]
        # self.X_test = self.X_data[train_size + validate_size:]
        # self.Y_test = self.Y_data[train_size + validate_size:]

    def sen_to_bag_of_words(self, sen):
        res = np.zeros(self.vocab_size)
        for word in sen:
            if self.word2index.__contains__(word):
                res[self.word2index[word]] += 1
        return res

    def sen_to_ngram(self, sen):
        res = np.zeros(self.ngram_size)
        res[:self.vocab_size] = self.sen_to_bag_of_words(sen)
        if len(sen) >= 3:
            for j in range(len(sen) - 2):
                temp = ' '.join(sen[j:j + 3])
                if self.ngram2index.__contains__(temp):
                    res[self.ngram2index[temp]] += 1
        elif len(sen) >= 2:
            for j in range(len(sen) - 1):
                temp = ' '.join(sen[j:j + 2])
                if self.ngram2index.__contains__(temp):
                    res[self.ngram2index[temp]] += 1
        return res

    def convert_to_onehot(self):
        Y = np.eye(self.classes)[list(self.Y_data)].T
        return Y

    def pre_process(self):
        df_train = pd.read_csv(self.path, sep='\t')
        # clean, tokenize and lemmatize
        df_train['Phrase'] = df_train['Phrase'].str.lower()
        df_train['Phrase'] = df_train['Phrase'].apply((lambda x: re.sub('[^a-zA-Z]', ' ', x)))
        lemmatizer = WordNetLemmatizer()
        words_list = []
        for sen in df_train.Phrase:
            words = word_tokenize(sen.lower())
            lemma_words = [lemmatizer.lemmatize(i) for i in words]
            words = []
            for i in lemma_words:
                if i not in stopwords_english:  # delete stopwords
                    words.append(i)
            words_list.append(words)
        self.X_data = np.array(words_list)
        self.Y_data = np.array(df_train.Sentiment)
        self.shuffle_data()
        self.Y_one_hot = self.convert_to_onehot()
        self.data_split()
        self.vocab = set()
        for i in self.X_data:
            for j in i:
                self.vocab.add(j)
        self.vocab_size = len(self.vocab)
        self.word2index = {}
        for index, value in enumerate(self.vocab):
            self.word2index[value] = index
        ngram_2 = dict()
        ngram_3 = dict()
        for tmp in self.X_data:
            if len(tmp) >= 3:
                for j in range(len(tmp) - 2):
                    trigram = ' '.join(tmp[j:j + 3])
                    if ngram_3.__contains__(trigram):
                        ngram_3[trigram] += 1
                    else:
                        ngram_3[trigram] = 1
            if len(tmp) >= 2:
                for j in range(len(tmp) - 1):
                    bigram = ' '.join(tmp[j:j + 2])
                    if ngram_2.__contains__(bigram):
                        ngram_2[bigram] += 1
                    else:
                        ngram_2[bigram] = 1
        keys = set(ngram_2.keys())
        for key in keys:
            if ngram_2[key] < 30:
                ngram_2.pop(key)
        keys = set(ngram_3.keys())
        for key in keys:
            if ngram_3[key] < 30:
                ngram_3.pop(key)
        self.ngram = np.concatenate(
            (np.array(list(self.vocab)), np.array(list(ngram_2.keys())), np.array(list(ngram_3.keys()))),
            axis=0)  # use vocab as unigram
        self.ngram_size = len(self.ngram)
        self.ngram2index = {}
        for index, value in enumerate(self.ngram):
            self.ngram2index[value] = index

    def get_bag_of_words(self, data):
        batch_size = len(data)
        res = np.empty([batch_size, self.vocab_size])
        for i in range(batch_size):
            res[i] = self.sen_to_bag_of_words(data[i])
        return res

    def get_n_gram(self, data):
        batch_size = len(data)
        res = np.empty([batch_size, self.ngram_size])
        for i in range(batch_size):
            res[i] = self.sen_to_ngram(data[i])
        return res
