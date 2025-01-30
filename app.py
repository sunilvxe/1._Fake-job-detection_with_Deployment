from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
with open("Random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(sentence):
    import string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    punc = string.punctuation
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    sentence1 = "".join([i for i in sentence.lower() if i not in punc])
    tokens = word_tokenize(sentence1)
    sentence2 = [i for i in tokens if i not in stop_words]
    sentence_preprocessed = " ".join([lemmatizer.lemmatize(i, "v") for i in sentence2])
    return sentence_preprocessed

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        vector = vectorizer.transform([processed_text])
        pred = model.predict(vector)
        prediction = "scam" if pred[0] == 1 else "not scam"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)