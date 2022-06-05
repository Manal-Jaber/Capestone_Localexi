import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

import logging

class EnglishArabicTranslator:

    def __init__(self, model_path, eng_tokenizer_path, ar_tokenizer_path):
        logging.info("EnglishArabicTranslator class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")

        # loading tokenizers
        with open(eng_tokenizer_path, 'rb') as handle:
            self.eng_tokenizer = pickle.load(handle)
        with open(ar_tokenizer_path, 'rb') as handle:
            self.ar_tokenizer = pickle.load(handle)
        logging.info("Tokenizers are loaded!")

    def predict(self, sentence):
        #From Data
        max_eng = 15
        max_fr = 21
        #Sentence preparation
        y_id_to_word = {value: key for key, value in self.fr_tokenizer.word_index.items()}
        y_id_to_word[0] = '<PAD>'
        sentence = self.eng_tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=max_eng, padding='post')

        # predict the class
        predictions = self.model.predict(sentence)
        result = (' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))

        # return class
        return result

def main():
	model = EnglishArabicTranslator('model.h5', 'eng_tokenizer.pickle', 'ar_tokenizer.pickle')
	predicted_class = model.predict("she is driving the truck")
	logging.info("This is the translation {}".format(predicted_class))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()