import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('feeling_clasification.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

labels = ['Sad/Sedih', 'Happy/Senang', 'Lover/Cinta', 'Angry/Marah', 'Scared/Takut', 'Suprised/terkejut']

def tebak_emosi(kalimat):
    sekuens = tokenizer.texts_to_sequences([kalimat])
    padded = pad_sequences(sekuens, maxlen=100, padding='post')
    
    prediksi = model.predict(padded, verbose=0)
    hasil_index = np.argmax(prediksi)
    
    print(f"\nKalimat: '{kalimat}'")
    print(f"Hasil Tebakan: {labels[hasil_index]} (Solid: {np.max(prediksi):.2f})")

while True:
    input_user = input("\nMasukkan Kalimat: ")
    if input_user.lower() == 'keluar':
        break
    tebak_emosi(input_user)