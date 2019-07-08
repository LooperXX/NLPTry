import re
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import collections
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity
from infersent import InferSent

stopwords_english = stopwords.words('english')
ngram_threshold = 2
embedding_size = 300

word2vec_path = 'E:\\workspace\\jupyter_notebook\\word_embeddings\\word2vec\\GoogleNews-vectors-negative300.bin'
glove_path_6B = 'E:\\workspace\\jupyter_notebook\\word_embeddings\\glove\\glove.6B.300d2word2vec.txt'  # glove.6B.300d
glove_path_840B = 'E:\\workspace\\jupyter_notebook\\word_embeddings\\glove\\glove.840B.300d2word2vec.txt'  # glove.840B.300d
glove_path_infer = 'E:\\workspace\\jupyter_notebook\\word_embeddings\\glove\\glove.840B.300d.txt'
fastText_path = 'E:\\workspace\\jupyter_notebook\\word_embeddings\\fastText\\crawl-300d-2M.vec'


class STSData:
    def __init__(self, path='data/input.txt', stop=True):
        self.path = path
        self.data = pd.read_csv(self.path, sep='\t')
        self.data['Sen1'] = self.preprocess(self.data['Sen1'], stop)
        self.data['Sen2'] = self.preprocess(self.data['Sen2'], stop)
        self.words_list = []
        self.words_list.extend(self.data['Sen1'])
        self.words_list.extend(self.data['Sen2'])
        self.X_data = np.array(self.words_list)
        self.num_sentence = int(self.X_data.shape[0] * 0.5)
        self.vocab = set()

    def preprocess(self, data, stop):
        # clean, tokenize and lemmatize
        data = data.str.lower()
        data = data.apply((lambda x: re.sub('[^a-zA-Z]', ' ', x)))
        lemmatizer = WordNetLemmatizer()
        words_list = []
        for sen in data:
            words = word_tokenize(sen.lower())
            lemma_words = [lemmatizer.lemmatize(i) for i in words]
            words = []
            if stop:
                for i in lemma_words:
                    if i not in stopwords_english:  # delete stopwords
                        words.append(i)
            else:
                for i in lemma_words:
                    words.append(i)
            words_list.append(words)
        return words_list

    def ngram_process(self):
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

        # 使用频度阈值对 Ngram 进行过滤
        keys = set(ngram_2.keys())
        for key in keys:
            if ngram_2[key] < ngram_threshold:
                ngram_2.pop(key)
        keys = set(ngram_3.keys())
        for key in keys:
            if ngram_3[key] < ngram_threshold:
                ngram_3.pop(key)

        self.ngram = np.concatenate(
            (np.array(list(self.vocab)), np.array(list(ngram_2.keys())), np.array(list(ngram_3.keys()))),
            axis=0)  # use vocab as unigram
        self.ngram_size = len(self.ngram)
        self.ngram2index = {}
        for index, value in enumerate(self.ngram):
            self.ngram2index[value] = index

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

    def sen_to_tf_idf(self, sen):
        # 词在句子中的频率 句子的总词数
        # 总句子数 包含该词的句子数
        res = np.zeros(self.vocab_size)
        fre = collections.Counter(sen)
        length = len(sen)
        # for word in sen:
        #     if fre.__contains__(word):
        #         fre[word] += 1
        #     else:
        #         fre[word] = 1

        for word in fre.keys():
            TF = fre[word] / length
            IDF = np.log(self.num_sentence / (self.word_in_corpus[word] + 1))
            res[self.word2index[word]] = TF * IDF
        return res

    # 词嵌入模型-基准方法 词嵌入直接平均
    def sen_to_embeddings_average(self, sen, embeddings):
        res = np.zeros((len(sen), embedding_size))
        for ix, word in enumerate(sen):
            if embeddings.__contains__(word):
                res[ix] = embeddings[word]
            else:
                res[ix] = embeddings['unk']
        res = np.mean(res, axis=0)
        return res

    # 词嵌入模型-基准方法 词嵌入 使用TF-IDF加权
    def sen_to_embeddings_average_tf_idf(self, sen, embeddings):
        res = np.zeros((len(sen), embedding_size))
        length = len(sen)
        tokfreqs = collections.Counter(sen)
        # length 归一化
        weights = [tokfreqs[token] / length * np.log(self.num_sentence / (self.word_in_corpus[token] + 1))
                   for token in tokfreqs]
        embedding = np.zeros((len(tokfreqs), embedding_size))
        for ix, token in enumerate(tokfreqs):
            if embeddings.__contains__(token):
                embedding[ix] = embeddings[token]
            else:
                embedding[ix] = embeddings['unk']
        res = np.average(embedding, axis=0, weights=weights)
        return res

    def get_bag_of_words(self, data, embedding):
        batch_size = len(data)
        res = np.empty([batch_size, self.vocab_size])
        for i in range(batch_size):
            res[i] = self.sen_to_bag_of_words(data[i])
        return res

    def get_n_gram(self, data, embedding):
        batch_size = len(data)
        res = np.empty([batch_size, self.ngram_size])
        for i in range(batch_size):
            res[i] = self.sen_to_ngram(data[i])
        return res

    def get_tf_idf(self, data, embedding):
        batch_size = len(data)
        res = np.empty([batch_size, self.vocab_size])
        for i in range(batch_size):
            res[i] = self.sen_to_tf_idf(data[i])
        return res

    def get_pretrain_embeddings(self, data, embedding):
        batch_size = len(data)
        res = np.empty([batch_size, embedding_size])
        for i in range(batch_size):
            res[i] = embedding['process_func'](data[i], embedding['embeddings'])
        return res

    def tf_idf_process(self):
        self.word_in_corpus = {}
        for word in self.vocab:
            self.word_in_corpus[word] = 0
            for sen in self.words_list:
                for word_ in sen:
                    if word == word_:
                        self.word_in_corpus[word] += 1
                        break


def analysis(data):
    length = []
    frequency = {}
    for sen in data.X_data:
        length.append(len(sen))
        for word in sen:
            if frequency.__contains__(word):
                frequency[word] += 1
            else:
                frequency[word] = 1
    # delete_words = []
    # for word in frequency.keys():
    #     if frequency[word] < 3:
    #         delete_words.append(word)
    length = collections.Counter(length)
    length_ = sorted(list(length.keys()), reverse=True)
    length_x = []
    length_y = []
    for leng in length_:
        length_x.append(leng)
        length_y.append(length[leng])

    plt.bar(range(len(length_y)), length_y, tick_label=length_x)
    plt.xlabel('Length Of Sentence')
    plt.ylabel('Frequency')
    plt.title('The Frequency Of The Length Of Sentence')
    plt.show()

    value = collections.Counter(frequency.values())
    value_ = sorted(list(value.keys()))
    value_x = []
    value_y = []
    for val in value_:
        if value[val] > 10:  # 为了绘图可见，只对词频值大于10的词频的频率进行可视化
            value_x.append(val)
            value_y.append(value[val])
        # print(val, value[val])
    plt.bar(range(len(value_y)), value_y, tick_label=value_x)
    plt.xlabel('Frequency Of Word')
    plt.ylabel('Frequency Of Word Frequency')
    plt.title('The Frequency Of Word Frequency')
    plt.show()
    # return delete_words


def calcMean(x, y):
    assert len(x) == len(y)
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x + 0.0) / n
    y_mean = float(sum_y + 0.0) / n
    return x_mean, y_mean


def calcPearson(x, y):
    x_mean, y_mean = calcMean(x, y)
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i] - x_mean) * (y[i] - y_mean)
    for i in range(n):
        x_pow += math.pow(x[i] - x_mean, 2)
    for i in range(n):
        y_pow += math.pow(y[i] - y_mean, 2)
    sumBottom = math.sqrt(x_pow * y_pow)
    p = sumTop / sumBottom
    return p


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = (0.5 + 0.5 * cos) * 5
    return sim


def predict(data, func, **embedding):
    features1 = func(data.Sen1, embedding)
    features2 = func(data.Sen2, embedding)
    sim = np.zeros(features1.shape[0])
    for i in range(features1.shape[0]):
        sim[i] = cos_sim(features1[i], features2[i])
    # sim = cosine_similarity(features1, features2)
    golden = np.zeros(features1.shape[0])
    with open('data/golden.txt') as f:
        index = 0
        for i in f:
            golden[index] = i
            index += 1
        # print(index)
    # return calcPearson(sim, golden)
    return np.corrcoef(sim, golden)[0, 1]


# 词嵌入模型-词移距离  Word Mover’s Distance  词移距离使用两文本间的词嵌入，测量其中一文本中的单词在语义空间中移动到另一文本单词所需要的最短距离。
def predict_wmd(data, embeddings):
    sim = np.zeros(data.Sen1.shape[0])
    index = 0
    for (sent1, sent2) in zip(data.Sen1, data.Sen2):
        sim[index] = embeddings.wmdistance(sent1, sent2)
        index += 1
    golden = np.zeros(data.Sen1.shape[0])
    with open('data/golden.txt') as f:
        index = 0
        for i in f:
            golden[index] = i
            index += 1
        # print(index)
    # return calcPearson(sim, golden)
    return np.corrcoef(sim, golden)[0, 1]


def predict_infersent(data, model_version):
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    MODEL_PATH = "encoder/infersent%s.pkl" % model_version
    model.load_state_dict(torch.load(MODEL_PATH))
    use_cuda = True
    model = model.cuda() if use_cuda else model
    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = glove_path_infer if model_version == 1 else fastText_path
    model.set_w2v_path(W2V_PATH)
    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=10000000)
    embeddings1 = model.encode(data.Sen1, bsize=128, tokenize=False, verbose=True)
    embeddings2 = model.encode(data.Sen2, bsize=128, tokenize=False, verbose=True)
    sim = cosine_similarity(embeddings1, embeddings2)
    golden = np.zeros(data.Sen1.shape[0])
    with open('data/golden.txt') as f:
        index = 0
        for i in f:
            golden[index] = i
            index += 1
        return np.corrcoef(sim, golden)[0, 1]


def test_Person_Correlation():
    test_A = np.random.randint(10, 50, size=(50,))
    test_B = np.random.randint(20, 60, size=(50,))
    r1 = np.corrcoef(test_A, test_B)
    r2 = calcPearson(test_A, test_B)
    print(r1[0, 1])
    print(r2)


if __name__ == '__main__':
    dataset = STSData()
    # analysis(dataset)
    dataset.ngram_process()
    print(dataset.vocab_size, dataset.ngram_size)

    #统计模型
    res = predict(dataset.data, dataset.get_bag_of_words)
    print('统计模型-词袋模型的皮尔逊相关系数值为：', res)
    res = predict(dataset.data, dataset.get_n_gram)
    print('统计模型-N-gram 模型的皮尔逊相关系数值为：', res, '使用 bi-gram & tri-gram，频率阈值为：', ngram_threshold)

    # ## N-gram 频率阈值实验
    # size = []
    # res_ = []
    # for i in range(1, 11):
    #     ngram_threshold = i
    #     dataset.ngram_process()
    #     print(dataset.vocab_size, dataset.ngram_size)
    #     res = predict(dataset.data, dataset.get_n_gram)
    #     size.append(dataset.ngram_size)
    #     res_.append(res)
    #     print('统计模型-N-gram 模型的皮尔逊相关系数值为：', res, '使用 bi-gram & tri-gram，频率阈值为：', ngram_threshold)
    #
    # plt.plot(range(len(size)), size)
    # plt.scatter(range(len(size)), size)
    # plt.xlabel('N-gram Frequency Threshold')
    # plt.ylabel('N-gram Embedding Size')
    # plt.xticks(list(range(11)), list(range(1, 12)))
    # plt.title('N-gram Embedding Size In Different N-gram Frequency Thresholds')
    # plt.show()
    #
    # plt.plot(range(len(res_)), res_)
    # plt.scatter(range(len(res_)), res_)
    # plt.xlabel('N-gram Frequency Threshold')
    # plt.ylabel('Person Correlation Value')
    # plt.xticks(list(range(11)), list(range(1, 12)))
    # plt.ylim(0.62, 0.7)
    # plt.title('Person Correlation Value In Different N-gram Frequency Thresholds')
    # plt.show()


    dataset.tf_idf_process()
    res = predict(dataset.data, dataset.get_tf_idf)
    print('统计模型-TF-IDF 模型的皮尔逊相关系数值为：', res)

    # test_Person_Correlation()

    # gensim glove to word2vec format
    # glove_file = datapath(glove_path)
    # tmp_file = get_tmpfile("glove.840B.300d2word2vec.txt")
    # _ = glove2word2vec(glove_file, tmp_file)
    # model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

    # 词嵌入模型
    embeddings_word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    embeddings_glove = gensim.models.KeyedVectors.load_word2vec_format(glove_path_840B)
    res_word2vec = predict(dataset.data, dataset.get_pretrain_embeddings, embeddings=embeddings_word2vec,
                           process_func=dataset.sen_to_embeddings_average)
    res_glove = predict(dataset.data, dataset.get_pretrain_embeddings, embeddings=embeddings_glove,
                        process_func=dataset.sen_to_embeddings_average)
    print('词嵌入模型-基准模型的皮尔逊相关系数值为： word2vec：', res_word2vec, '\t glove：', res_glove)
    dataset.tf_idf_process()
    res_word2vec = predict(dataset.data, dataset.get_pretrain_embeddings, embeddings=embeddings_word2vec,
                           process_func=dataset.sen_to_embeddings_average_tf_idf)
    res_glove = predict(dataset.data, dataset.get_pretrain_embeddings, embeddings=embeddings_glove,
                        process_func=dataset.sen_to_embeddings_average_tf_idf)
    print('词嵌入模型-基准模型-TF_IDF加权的皮尔逊相关系数值为： word2vec：', res_word2vec, '\t glove：', res_glove)
    res_word2vec = predict_wmd(dataset.data, embeddings_word2vec)
    res_glove = predict_wmd(dataset.data, embeddings_glove)
    print('词嵌入模型-词移距离的皮尔逊相关系数值为： word2vec：', res_word2vec, '\t glove：', res_glove)

    # 词嵌入模型-infersent 预训练完毕的句子级encoder
    data1 = []
    data2 = []
    for sen in dataset.data.Sen1:
        data1.append(' '.join(sen))
    for sen in dataset.data.Sen2:
        data2.append(' '.join(sen))
    dataset.data.Sen1 = data1
    dataset.data.Sen2 = data2

    model_version = 1 # V1 trained with GloVe, V2 trained with fastText
    res_glove = predict_infersent(dataset.data, model_version)
    model_version = 2
    res_fastText = predict_infersent(dataset.data, model_version)
    print('词嵌入模型-infersent 的皮尔逊相关系数值为： glove：', res_glove, '\t fastText：', res_fastText)


    # # bert-serving-start -model_dir E:\\workspace\\jupyter_notebook\\word_embeddings\\Bert\\uncased_L-12_H-768_A-12\\ -num_worker=4
    # from bert_serving.client import BertClient
    # # dataset = STSData()
    # # data = list(dataset.data.Sen1)
    # # data.append(list(dataset.data.Sen2))
    # bc = BertClient()
    # bc.encode(['First do it', 'then do it right', 'then do it better'])
