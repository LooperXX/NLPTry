import pickle
import numpy as np
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import SAOMR
stopwords_english = stopwords.words('english')
np.random.seed(7)


def initialize_parameters(dim):
    n_x, n_h, n_y = dim[0], dim[1], dim[2]
    np.random.seed(1)
    # 初始化
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0)
    A = np.exp(Z_shift) / np.sum(np.exp(Z_shift), axis=0)
    cache = Z_shift
    return A, cache


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    A, linear_cache, activation_cache = None, None, None
    if activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def compute_loss(Y_hat, Y):
    batch_size = Y.shape[1]
    loss = -(np.sum(Y * np.log(Y_hat))) / float(batch_size)
    assert (loss.shape == ())
    return loss


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_, W, b = cache
    batch_size = A_.shape[1]

    dW = np.dot(dZ, A_.T) / float(batch_size)
    db = np.sum(dZ, axis=1, keepdims=True) / float(batch_size)
    dA_ = np.dot(W.T, dZ)

    assert (dA_.shape == A_.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_, dW, db


def softmax_backward(Y, cache):
    Z = cache
    s = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    dZ = s - Y
    assert (dZ.shape == Z.shape)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    dA_, dW, db = None, None, None
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_, dW, db = linear_backward(dZ, linear_cache)

    return dA_, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters


def predict_labels(X, Y, parameters, feature):
    batch_size = X.shape[0]
    X = feature(X).transpose(1, 0)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Forward propagation
    A1, _ = linear_activation_forward(X, W1, b1, activation='relu')
    probs, _ = linear_activation_forward(A1, W2, b2, activation='softmax')

    # convert probas to 0-9 predictions
    predict_label = np.argmax(probs, axis=0)
    accuracy = 0
    if Y is not None:
        accuracy = np.sum((predict_label == Y) / float(batch_size))

    return predict_label, accuracy


def two_layer_model(X, Y, X_v, Y_v, layers_dims, learning_rate, epochs, feature, print_cost=False, batch_size=None, ):
    grads = {}
    losses = []
    accuracies = []
    if batch_size is None:
        batch_size = X.shape[0]
    shuffle_index = list(range(X.shape[0]))
    times = X.shape[0] // batch_size
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, epochs):
        loss = 0
        for t in tqdm(range(times)):
            index = shuffle_index[batch_size * t:batch_size * (t + 1)]
            X_ = feature(X[index]).transpose(1, 0)
            Y_ = Y[:, index]
            # Forward propagation
            A1, cache1 = linear_activation_forward(X_, W1, b1, activation='relu')
            A2, cache2 = linear_activation_forward(A1, W2, b2, activation='softmax')
            # Compute cost
            loss += compute_loss(A2, Y_)
            # Backward propagation
            dA1, dW2, db2 = linear_activation_backward(Y_, cache2, activation='softmax')
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
        loss /= times
        losses.append(loss)
        _, accuracy = predict_labels(X_v, Y_v, parameters, feature)
        accuracies.append(accuracy)
        print("epoch {}\tloss {}\tval_accuracy {}".format(i + 1, loss, accuracy))
    # plot cost
    plt.plot(np.squeeze(losses))
    plt.ylabel('loss')
    plt.xlabel('iterations ')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def save_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_test(path, parameters, dataset, feature):
    df_train = pd.read_csv(path, sep='\t')
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
        words_list.append(lemma_words)
    X_data = np.array(words_list)
    predict_label, _ = predict_labels(X_data, None, parameters, feature)
    sub_file = pd.read_csv('data/sampleSubmission.csv',sep=',')
    sub_file.Sentiment = predict_label
    sub_file.to_csv('Submission.csv',index=False)

def main():
    # batch32_bagofwords_shuffle_lr0.01_epoch100_hidden128
    # batch [1,16,32,64,128,256]
    # feature bag_of_words / n_gram
    # shuffle True / false

    shuffle = True
    # dataset = SAOMR(shuffle=shuffle)
    # save_file(dataset, 'dataset.pkl')
    dataset = load_file('dataset.pkl')
    learning_rate = 0.1
    epochs = 100
    hidden = 128
    batch_size = 32  # max 124848(BGD) 1(SGD) 16,32,64,128,256(mini-batch)
    # feature = dataset.get_bag_of_words
    # layers_dims = (dataset.vocab_size, hidden, 5)
    feature = dataset.get_n_gram
    layers_dims = (dataset.ngram_size, hidden, 5)
    parameters = two_layer_model(dataset.X_train, dataset.Y_train, dataset.X_validate, dataset.Y_validate, layers_dims,
                                 learning_rate, epochs, feature, print_cost=True, batch_size=batch_size)
    # accuracy = predict_labels(dataset.X_test, dataset.Y_test, parameters, feature)
    # print('test_accuracy', accuracy)
    predict_test('test.tsv', parameters, dataset, feature)



    parameters = two_layer_model(dataset.X_train, dataset.Y_train, dataset.X_validate, dataset.Y_validate, layers_dims,
                                 learning_rate, epochs, feature, print_cost=True, batch_size=batch_size)
    # accuracy = predict_labels(dataset.X_test, dataset.Y_test, parameters, feature)
    # print('test_accuracy', accuracy)
    predict_test('../input/test.tsv', parameters, dataset, feature)
if __name__ == "__main__":
    main()
