from keras.models import Sequential,Model
from keras.preprocessing import sequence,text
from keras.layers import Embedding, Dense, LSTM,Bidirectional,Dropout,Input,GlobalMaxPool1D
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

""" Load Data and sample submission file """
train_data = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/train/train.csv')
test_data = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/test/test.csv')
submission = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/sample_submission.csv')


Max_features = 30000
maxlen = 200
embed_size = 50

list_train_data = train_data.comment_text.fillna('missing text').values
list_test_data = test_data.comment_text.fillna('missing text').values
label = train_data.columns[2:]
y = train_data[label].values

tokenizer = text.Tokenizer(num_words=Max_features)
tokenizer.fit_on_texts(list(list_train_data))
list_tokenized_train = tokenizer.texts_to_sequences(list_train_data)
x_train = sequence.pad_sequences(list_tokenized_train,maxlen=maxlen)
list_tokenized_test = tokenizer.texts_to_sequences(list_test_data)
x_test = sequence.pad_sequences(list_tokenized_test,maxlen=maxlen)

vocab_size = len(tokenizer.word_index)+1



with open('/home/nikit/Desktop/Glove_word_vectos/glove.twitter.27B.50d.txt') as glove_twitter:
    embedding_index = dict()
    for line in glove_twitter:
        value = line.split()
        word = value[0]
        vector = np.asarray(value[1:],dtype="float32")
        embedding_index[word] = vector
    glove_twitter.close()
#embeddings = np.stack(embedding_index.values())
nb_words = min(Max_features,vocab_size)
embedding_matrix = np.random.normal(0.0209404, 0.6441043, (nb_words, embed_size))
for word,i in tokenizer.word_index.items():
    if i>= Max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def get_model():
    embed_size = 50
    inp = Input(shape=(maxlen, ))
    x = Embedding(Max_features, embed_size,weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()
batch_size = 32
epochs = 1
save_parameter_file_path = '/home/nikit/Desktop/Kaggle/toxic_comments/weights.best.hdf5'
checkpoint = ModelCheckpoint(save_parameter_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


callbacks_list = [checkpoint, early] #early

model.fit(x_train, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

model.load_weights(save_parameter_file_path)

y_test = model.predict(x_test)

submission[label] = y_test
submission.to_csv('/home/nikit/Desktop/Kaggle/toxic_comments/result_keras1.csv',index=False)
