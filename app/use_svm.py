import os
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from flask import Flask, request

# NLTK data
nltk.download('stopwords')
nltk.download('punkt')


# Init model
model = None
# Init embeddings dictionary
embeddings_dict = {}
# Init flask app
app = Flask(__name__)


# Load model function
def load_model():
    global model    
    model = pickle.load(open('final_svm.pkl', 'rb'))    # Load saved model through pickle


# GloVe Embeddings
def glove():
    global embeddings_dict
    # Create Embeddings Dictionary (maps words to vectors )
    with open('glove.6B.100d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


# Prepare data function
def prepare_data(data):

    # Set up text processing
    stop_words = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop_words.update(punctuation)
    stop_words.update('‘')
    stop_words.update('’')

    # Process data: delete whitespaces, everything lowercase, tokenize data, and delete stopwords
    data = data.strip().lower()
    data = word_tokenize(data)
    data = np.array([w.strip() for w in data if not w.strip().lower() in stop_words])

    # Transfor each word to a vector
    M = []
    for w in data:
        try:
            M.append(embeddings_dict[w])
        except:
            continue

    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(100)
    return v / np.sqrt((v ** 2).sum())



# Main route set up
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return 'Welcome! Now you can make new predictions!'

    if request.method == 'POST':
        # Get data
        input_data = str(request.get_json())
        # PROCESS DATA
        input_data = prepare_data(input_data)   # Text processing
        input_data = np.array(input_data)       # Transfor the data into a numpy array
        input_data = input_data.reshape(1, -1)  # Reshape the data
        # Make prediction with model
        response = model.predict(input_data)
    # Return predictions
    if response[0] == 0:
        return 'This is probably a fake news'   # Fake news prediction
    elif response[0] == 1:
        return 'This is probably a real news'   # Real news prediction
    else:
        return 'Something went wrong'           # Error




if __name__ == '__main__':
    # Load model
    print('Loading model...')
    load_model()
    # Create embeddings dictionary
    print('Creating word embeddings dictionary...')
    glove()
    # Run app
    port = int(os.environ.get('PORT', 5000))    # Get PORT in env or set it to 5000
    app.run(host='0.0.0.0', port=port)
