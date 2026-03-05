import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout, Reshape, Bidirectional

MODEL_PATH = 'feeling_clasification.keras'
TOKENIZER_PATH = 'tokenizer.pickle'
MAX_WORDS = 20000
MAX_LEN = 100

def load_and_combine():
    df_train_en, df_train_id, df_train_id_plus, df_train_shorted, df_train_repair = pd.read_csv('training_en_shorted.csv'), pd.read_csv('train_id.csv'), pd.read_csv('train_id_sadsurprise.csv'), pd.read_csv('emotion_multilingual.csv'), pd.read_csv('emotion_multilingual_repair.csv')
    df_val = pd.read_csv('validation_multi.csv')

    df_combined = pd.concat([df_train_en, df_train_id, df_train_id_plus, df_train_shorted, df_train_repair], ignore_index=True)
    df_all = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_all, df_val
df_all, df_val = load_and_combine()

if os.path.exists(TOKENIZER_PATH):
    print("--- Loading Tokenizer lawas... ---")
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    print("--- Nggawe Tokenizer anyar... ---")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df_all['text'])
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train = pad_sequences(tokenizer.texts_to_sequences(df_all['text']), maxlen=MAX_LEN, padding='post')
y_train = df_all['label'].values
X_evaluate = pad_sequences(tokenizer.texts_to_sequences(df_val['text']), maxlen=MAX_LEN, padding='post')
y_evaluate = df_val['label'].values

if os.path.exists(MODEL_PATH):
    print(f"--- {MODEL_PATH} Ketemu! Loading model kanggo Retrain... ---")
    model = load_model(MODEL_PATH)
else:
    print("--- Model gak ketemu. Nggawe Arsitektur anyar... ---")
    model = tf.keras.Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Reshape((1, 128)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.7),
        Dense(6, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\n--- Eksekusi Training... ---")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose = 1, callbacks=[early_stop], validation_data=(X_evaluate, y_evaluate))

model.save(MODEL_PATH)
print(f"--- Sukses! Model diupdate lan disimpen ing {MODEL_PATH} ---")