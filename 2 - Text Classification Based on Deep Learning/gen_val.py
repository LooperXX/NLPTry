import numpy as np
import pandas as pd


def save_csv(data, name, columns):
    save = pd.DataFrame(data, columns=columns)
    save.to_csv(name, index=False)


df_train = pd.read_csv('data/train.tsv', sep='\t')
df_test = pd.read_csv('data/test.tsv', sep='\t')
data = np.array([df_train.Phrase, df_train.Sentiment]).transpose(1, 0)
np.random.shuffle(data)
train_size = int(len(data) * 0.8)
data_train, data_val = data[:train_size], data[train_size:]
data_test = np.array([df_test.Phrase, df_test.PhraseId]).transpose(1, 0)
save_csv(data_train, 'train.csv', ['text', 'label'])
save_csv(data_val, 'val.csv', ['text', 'label'])
save_csv(data_test, 'test.csv', ['text', 'label'])
