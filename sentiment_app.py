#------------------------------------- IMPORTS ---------------------------------------
import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

app = FastAPI()

#------------------------------------ HELPER FUNCTIONS -------------------------------
# method for cleaning input, so it's the same format as the tweets we trained on
def preProcess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

data = pd.read_csv('Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)

def my_pipeline(text): #pipeline
    text_new = preProcess_data(text)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X, maxlen=28)
    return X
#-------------------------------------------------------------------------------------

# create a dummy route on the home-page
@app.get('/') #basic get view, tells the user how to use the app
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post, or send post request to /predict "}

# method for creating input form for user generated input (test string)
# uses HTMLResponse from starlette.response
@app.get('/predict', response_class=HTMLResponse) # get from input form
def take_inp():
    return '''<form method="post">
 <input type="text" maxlength="28" name="text" value="Input text to be tested"/>
 <input type="submit"/>
 </form>'''

# POST request, takes data from input form and passes into model for prediction
@app.post('/predict') # prediction on data
def predict(text:str = Form(...)): # take input from forms
    clean_text = my_pipeline(text) # clean and preprocess the texts
    loaded_model = tf.keras.models.load_model('py_sentiment.h5') # load saved model
    predictions = loaded_model.predict(clean_text) # predict
    sentiment = int(np.argmax(predictions)) # index of max prediction
    probability = max(predictions.tolist()[0]) # probability of max prediction
    if sentiment==0: # assign "name" to prediction
        t_sentiment = 'negative'
    elif sentiment==1:
        t_sentiment = 'neutral'
    elif sentiment==2:
        t_sentiment='postive'

    return { # returns a dictionary as endpoint
        "Input Sentence": text,
        "Predicted Sentiment": t_sentiment,
        "Probability": probability
     }
