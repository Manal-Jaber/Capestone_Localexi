import os
import logging

from flask import Flask, request, jsonify, render_template

from model.model import EnglishArabicTranslator

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


if __name__ == "__main__":
    main()