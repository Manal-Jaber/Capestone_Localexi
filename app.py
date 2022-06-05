import os
import logging

import pandas as pd
from flask import Flask, request, jsonify, render_template, request

from model.model import EnglishArabicTranslator

import srt

app = Flask(__name__, template_folder='templates', static_folder='static')

# define model path
model_path = './model/model.h5'
eng_tokenizer_path = './model/eng_tokenizer.pickle'
ar_tokenizer_path = './model/ar_tokenizer.pickle'

# create instance
model = EnglishArabicTranslator(model_path, eng_tokenizer_path, ar_tokenizer_path)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    """Provide simple health check route."""
    return render_template('index.html')

@app.route('/result', methods = ['GET', 'POST'])
def upload_file():
   logging.Logger(request)
   if request.method == 'POST':
      f = request.files['english-file']
      file_name = f.filename
      f.save(file_name)
      df = create_dataframe_single_file(file_name)
      return df

@app.route("/docs")
def docs():
    """Provide simple health check route."""
    return render_template('docs.html')


@app.route("/v1/translate", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")
    sentence = request.args.get("sentence")
    prediction = model.predict(sentence)

    logging.info("translation from model= {}".format(prediction))
    return jsonify({"translated_sentence": str(prediction)})

def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8000))
, debug=True)


# Defining a function that opens srt files
def load_srt(filename):
    print(filename)
    # parse .srt file to list of subtitles
    print("Loading {}".format(filename))
    with open(filename) as f:
        text = f.read()
    return list(srt.parse(text))


# Defining a function that parse the data file into a dataframe
def create_dataframe_single_file(file_name):
    name, ext = os.path.splitext(file_name)
    if(ext.lower() == '.srt'):
        data=[]
        dataList = load_srt(file_name)
        for dataLine in dataList:
            start = dataLine.start
            end = dataLine.end
            content = dataLine.content
            data.append([start, end, content])
        df = pd.DataFrame(data)
        df.columns=['StartTime', 'EndTime', 'Content']
        return df
    else:
        return 'Wrong File Type'

if __name__ == "__main__":
    main()