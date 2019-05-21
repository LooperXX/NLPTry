# %%
import time

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from torchtext.data import Field, Iterator, BucketIterator, TabularDataset
from torchtext.vocab import Vectors

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
data_path = 'E:\\workspace\\python-workspace\\NLPTry\\2 - Text Classification Based on Deep Learning\\'
vocab_path = 'E:\\workspace\\jupyter_notebook\\.vector_cache\\'
load_model_path = None
classes = 5
max_len = 56
hidden_dim = 100
num_layers = 2
freeze_embeddings = True
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
class LSTMBase(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_dim, classes, hidden_dim, batch_size, num_layers=2, freeze_embeddings=True, prob=0.2):
        super(LSTMBase, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embeddings, requires_grad=freeze_embeddings)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=num_layers, dropout=prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, classes)
        self.init = self.init_hidden()

    def init_hidden(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        h0 = t.zeros((self.num_layers, batch_size, self.hidden_dim))
        c0 = t.zeros((self.num_layers, batch_size, self.hidden_dim))
        if use_gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0

    def forward(self, input):
        # (batch_size, seq_length, embedding_dim)
        embeddings = self.embedding(input)
        h, c = self.init_hidden(embeddings.size()[0])
        out, (h, c) = self.lstm(embeddings, (h, c))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return F.softmax(out, dim=0)

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/TC_LSTM' + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        t.save(self.state_dict(), name)
        return name

# %%
model = LSTMBase(weight_matrix, weight_matrix.size(0), weight_matrix.size(1), classes, hidden_dim, batch_size, num_layers, freeze_embeddings)
if load_model_path is not None:
    model.load(load_model_path)
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
        model.save()

# %%
# train(model, train_dl, valid_dl, epochs, print_every=print_every)

# %%
test_losses = []  # track loss
num_correct = 0
model.eval()
# iterate over test data
res = np.empty([len(test_dl.dl.dataset), 2])
index = 0
for inputs, labels in test_dl:
    if use_gpu:
        model.cuda()
        inputs, labels = inputs.cuda(), labels.cuda()
    output = model(inputs.transpose(1, 0))
    predict_label = np.argmax(output.detach().cpu(), axis=1)
    num = len(predict_label)
    res[index:index + num] = t.stack((labels.squeeze().detach().cpu(), predict_label.float()), 0).transpose(1, 0)
    index += num
# print(index, len(test_dl.dl.dataset))
