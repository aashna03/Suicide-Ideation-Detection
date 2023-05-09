# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import nltk
nltk.download("punkt")
from nltk.corpus import stopwords
nltk.download("stopwords")

# %%
nltk.download('omw-1.4')
nltk.download('wordnet')

# %%
import re
tokenizer = nltk.RegexpTokenizer(r"[A-Za-z]\w+")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv('Suicide_Detection.csv',encoding='ISO-8859-1',on_bad_lines='skip')
data.head()

# %%
data = data.iloc[:, 1:3]
data.head()

# %%
data['class'].value_counts()

# %%
data = data[data['class'].apply(lambda x: x=="suicide" or x=="non-suicide")]

# %%
data['class'].value_counts()

# %%
data = data.dropna()
data.info()

# %%
data['class'] = data['class'].replace('non-suicide', 0)
data['class'] = data['class'].replace('suicide', 1)

# %%
data.head()

# %%
data = data[data['text'].apply(lambda x: len(x.split())<=170)]
data.reset_index(drop=True, inplace=True)

# %%
data['class'].value_counts()


# %%
df_grouped_by = data.groupby(['class'])
df_balanced = df_grouped_by.apply(lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True))
df_balanced = df_balanced.droplevel(['class'])
df_balanced

# %%
from collections import Counter

# %%
pip install imbalanced-learn

# %%
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)

x,y = over_sampler.fit_resample(data['text'].values.reshape(-1,1), data['class'])
print(f"Training target statistics: {Counter(y)}")

# %%
x[:5]

# %%
x = x.flatten()
x.shape

# %%
for i in range(0,216612):
    tokens = tokenizer.tokenize(x[i])
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    token_update = [word for word in lemmatized_tokens if not word in stopwords.words()]
    x[i] = (" ").join(token_update)
    if i%500 == 0:
      print(i," ",x[i])


# %%
temp_x = x
temp_y = y

# %%
y.tolist()

# %%
x.tolist()

# %%
data = pd.DataFrame([x,y]).transpose()
data.head()

# %%
data.columns = ['text', 'label']

# %%
data.head()

# %%
data.to_csv('clean_textual.csv')

# %%
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

# %%
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from collections import Counter
import matplotlib.pyplot as plt

# %%
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 0

# %%
data = pd.read_csv('clean_textual.csv')
data.head()

# %%
data = data.iloc[:,1:]

# %%
data.head()

# %%
data['label'].value_counts()


# %%
train_text, temp_text, train_labels, temp_labels = train_test_split(data['text'], data['label'],
                                                                    random_state=SEED,
                                                                    test_size=0.6,
                                                                    stratify=data['label'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=SEED,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

data.text=data.text.astype(str)


# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)
vocab_size = len(tokenizer.word_index) + 1

# %%
vocab_size


# %%
vocab = Counter()
tokens_list = [(s.split()) for s in train_text]
for i in tokens_list:
  vocab.update(i)
min_occurance = 2
tokens = [k for k,c in vocab.items() if c >= min_occurance]
print(len(tokens))

# %%
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
save_list(vocab, 'vocab.txt')

# %%
def clean_line(line, vocab):
  tokens = line.split()
  tokens_clean = [w for w in tokens if w in vocab]
  return [tokens_clean]

def process_lines(data, vocab):
  lines = list()
  for i in data:
    line = clean_line(i, vocab)
    lines += line
  return lines

# %%
train_clean = process_lines(train_text, vocab)
test_clean = process_lines(test_text, vocab)

# %%
model = Word2Vec(vector_size=100, window=4, min_count=2, epochs=18, seed=SEED)

# %%
model.build_vocab(train_clean, progress_per=200)

# %%
model.train(train_clean, total_examples=model.corpus_count, epochs=EPOCHS,report_delay=1)

# %%
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

# %%

model.wv.most_similar('suicide')
# %%
def tokenize_and_encode(text, max_length=70):
    encoded_docs = tokenizer.texts_to_sequences(text)
    padded_sequence = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_sequence

tokens_train = tokenize_and_encode(train_text)
tokens_val = tokenize_and_encode(val_text)
tokens_test = tokenize_and_encode(test_text)

# %%

def load_embedding(filename):
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	embedding = dict()
	for line in lines:
		parts = line.split()
		embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
	return embedding
# %%
def get_weight_matrix(embedding, vocab, embedding_dim):
	vocab_size = len(vocab) + 1
	weight_matrix = np.zeros((vocab_size, embedding_dim))
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix

# %%

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
# %%

raw_embedding_word2vec = load_embedding('embedding_word2vec.txt') 
embedding_vectors_word2vec = get_weight_matrix(raw_embedding_word2vec, tokenizer.word_index, 100)
embedding_vectors_word2vec = np.float32(embedding_vectors_word2vec)
# %%

from keras.layers import Embedding,Dense,LSTM,Bidirectional,GlobalMaxPooling1D,Input,Dropout,Conv1D,MaxPooling1D,Flatten
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
 
# %%
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=70))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# %%
train=model.fit(tokens_train,train_labels,validation_data=(tokens_val,val_labels),epochs=4,batch_size=256)

# %%
plt.figure(figsize=(10,8))
plt.plot(train.history['accuracy'])
plt.plot(train.history['val_accuracy'])
plt.title('ACCURACY CURVE',fontdict={'size':20})
plt.show()

# %%
test_predictions = (model.predict(tokens_test) > 0.5).astype(int)
from sklearn.metrics import classification_report
print(classification_report(test_labels, test_predictions))
test_predictions.shape

# %%
model1 = Sequential()
model1.add(Embedding(vocab_size, 100, input_length=70))
model1.add(LSTM(units=100,return_sequences = True))
model1.add(Dense(10))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())

# %%
train=model1.fit(tokens_train,train_labels,validation_data=(tokens_val,val_labels),epochs=5,batch_size=256)

# %%
plt.figure(figsize=(7,7))
plt.plot(train.history['accuracy'])
plt.plot(train.history['val_accuracy'])
plt.title('ACCURACY CURVE',fontdict={'size':20})
plt.show()
# %%
test_predictions = model1.predict(tokens_test)
pred = (test_predictions.sum(axis=1)/70 > 0.5).astype(int).flatten()
print(classification_report(test_labels, pred))
