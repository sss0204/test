import re
import urllib.request
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer

target_text = open('mouse_reviews.txt', 'r', encoding= "utf-8")
train_data = pd.read_table('mouse_reviews.txt')
train_data[:5]
print(len(train_data))
print(train_data.isnull().values.any())
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
train_data['review'] = train_data['review'].str.replace("^ㄱ-ㅎㅏ-ㅣ가-힣", "")
train_data[:5]

stopwords = ['의', '이', '가', '은', '들', '는', '좀', '잘', '강', '과', '도', '를','걍', '입니다', '으로', '자', '에', '하다', '이다']


okt = Okt()
tokenized_data = []
for sentence in train_data['review']:
    temp_x = okt.morphs(sentence, stem = True)
    temp_x = [word for word in temp_x if not word in stopwords]
    tokenized_data.append(temp_x)

tokenized_data    

print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_data)
print(tokenizer.word_index)
threshold = 3
total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    
    if(value < threshold):
        rare_cnt = rare_cnt +1
        rare_freq = rare_freq + value
        
print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt +1
print('단어 집합의 크기: ', vocab_size)
tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(tokenized_data)
X_train = tokenizer.texts_to_sequences(tokenized_data)
print(X_train[:3])

from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg=1)

model.wv.save_word2vec_format('mouse_w2v') # 모델 저장

from gensim.models import KeyedVectors
loaded_model = KeyedVectors.load_word2vec_format("mouse_w2v") # 모델 로드

model_result = loaded_model.most_similar("소음")
print(model_result)

import numpy as np
y_train = np.array(train_data['score'])
y_train = [0 if i <4 else 1 for i in y_train]
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))

print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, X_train)

from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, maxlen = max_len)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
