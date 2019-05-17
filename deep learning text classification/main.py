# %%
import re
import os
import numpy as np
import pandas as pd
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import Field, Iterator, BucketIterator, TabularDataset
import torch as t
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stopwords_english = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
# max_len = 0
# for i in train.examples:
#     max_len = max(max_len, len(vars(i)['Text']))
# for i in val.examples:
#     max_len = max(max_len, len(vars(i)['Text']))
# for i in test.examples:
#     max_len = max(max_len, len(vars(i)['Text']))
# max_len
# %%
data_path = 'E:\\workspace\\python-workspace\\NLPTry\\deep learning text classification\\'
vocab_path = 'E:\\workspace\\jupyter_notebook\\.vector_cache\\'
classes = 5
max_len = 56
num_filters = 100
kernel_sizes = [3, 4, 5]
lr = 0.001
batch_size = 32
epochs = 10
print_every = 100
device = t.device('cuda:0')
use_gpu = True


# %%
def tokenize_en(text):
    words = word_tokenize(text)
    return [lemmatizer.lemmatize(i) for i in words]


# %%
TEXT = Field(tokenize=tokenize_en,
             fix_length=max_len, stop_words=stopwords_english,
             lower=True)
LABEL = Field(sequential=False, use_vocab=False)
# %%
train, val, test = TabularDataset.splits(
    path=data_path, train='train.csv', skip_header=True,
    validation='val.csv', test='test.csv', format='csv',
    fields=[('text', TEXT), ('label', LABEL)])
# %%
vector = Vectors("glove.6B.100d.txt", cache=vocab_path)
# vector.unk_init = init.xavier_uniform
TEXT.build_vocab(train, vectors=vector)
weight_matrix = TEXT.vocab.vectors
weight_matrix = weight_matrix.cuda()
# %%
train_iter, val_iter = BucketIterator.splits(
    (train, val),
    batch_sizes=(batch_size, batch_size),
    device=device,
    sort_key=lambda x: len(x.Text),
    # the BucketIterator needs to be told what function it should use to group the data.
    sort_within_batch=False
)
test_iter = Iterator(test, batch_size=batch_size, device=device, sort=False, sort_within_batch=False, repeat=False)

# %%
class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is not None:  # we will concatenate y into a single tensor
                y = t.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = t.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


# %%
train_dl = BatchWrapper(train_iter, "text", ["label"])
valid_dl = BatchWrapper(val_iter, "text", ["label"])
test_dl = BatchWrapper(test_iter, "text", ["label"])


# next(train_dl.__iter__())
# %%
class TextCNN(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_dim, output_size,
                 num_filters=100, kernel_sizes=None, freeze_embeddings=True, drop_prob=0.5):
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        super(TextCNN, self).__init__()
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embeddings)  # all vectors
        if freeze_embeddings:
            self.embedding.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim))
            for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)
        self.dropout = nn.Dropout(drop_prob)
        self.softmax = nn.Softmax()

    def conv_and_pool(self, x, conv):
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length, 1) -> (batch_size, num_filters, conv_seq_length)
        x = F.relu(conv(x)).squeeze()
        # 1D pool over conv_seq_length, squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        # (batch_size, seq_length, embedding_dim)
        embeddings = self.embedding(x)
        # embeddings.unsqueeze(1) creates a channel dimension that conv layers expect
        # (batch_size, channel, seq_length, embedding_dim)
        embeddings = embeddings.unsqueeze(1)
        conv_results = [self.conv_and_pool(embeddings, conv) for conv in self.convs]
        # concatenate results
        x = t.cat(conv_results, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return self.softmax(logits)


# %%
model = TextCNN(weight_matrix, weight_matrix.size(0), weight_matrix.size(1), classes, num_filters, kernel_sizes)
print(model)
# %%
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=lr)


# %%
# training loop
def train(model, train_loader, valid_loader, epochs, print_every=100):
    if use_gpu:
        model.cuda()
    counter = 0
    model.train()
    for e in range(epochs):
        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            model.zero_grad()
            output = model(inputs)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            if counter % print_every == 0:
                val_losses = []
                accuracy = []
                model.eval()
                for inputs, labels in valid_loader:
                    if (use_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()
                    output = model(inputs)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())
                    predict_label = np.argmax(output, axis=0)
                    accuracy.append(np.sum((predict_label == labels) / float(batch_size)))
                model.train()
                print("Epoch\t{}/{}...".format(e + 1, epochs),
                      "Step\t}...".format(counter),
                      "Loss\t{:.6f}...".format(loss.item()),
                      "Val_Loss\t{:.6f}".format(np.mean(val_losses)),
                      "Val_Accuracy\t{:.6f}...".format(accuracy))


# %%
# train(model, train_dl, valid_dl, epochs, print_every=print_every)
# %%
test_losses = []  # track loss
num_correct = 0
model.eval()
# iterate over test data
res = np.empty([len(test_dl.dl.dataset), 2])
for inputs, labels in test_dl:
    if use_gpu:
        model.cuda()
        inputs, labels = inputs.cuda(), labels.cuda()
    output = model(inputs.transpose(1, 0))
    predict_label = np.argmax(output, axis=0)
    t.cat([labels, predict_label], 0)
    break
# test_losses = [] # track loss
# num_correct = 0
# model.eval()
# # iterate over test data
# for inputs, labels in test_dl:
#
#     if(use_gpu):
#         inputs, labels = inputs.cuda(), labels.cuda()
#     output = model(inputs)
#     test_loss = criterion(output.squeeze(), labels.float())
#     test_losses.append(test_loss.item())
#     predict_label = np.argmax(output, axis=0)
#     # compare predictions to true label
#     correct = np.sum(predict_label == labels)
#     num_correct += correct
#
# print("Test loss\t{:.6f}".format(np.mean(test_losses)))
# test_acc = num_correct/len(test_dl.dl.dataset)
# print("Test accuracy\t{:.3f}".format(test_acc))
# %%

