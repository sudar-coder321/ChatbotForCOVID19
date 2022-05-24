from flask import Flask,render_template,request

#    test_questions = [
#         "What are rules to be followed?",
#     # "What about pregnant women?",
#     # "Wat is de lengte van de incubatietijd?",
#     # "Are animals contagious COVID-19?",
#     # "Are there medicine against the coronavirus?",
#     # "Can I breastfead when I have COVID-19?",
#     # "Should I stay inside the house?",  # English questions are also possible.
#     # "Kann ich mit meinem Hund spazieren gehen?",# As well as German, and all the other languages supported by use-multilingual.
#         ]


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import re

def preprocess_sentences(input_sentences):
    return [re.sub(r'(covid-19|covid)', 'coronavirus', input_sentence, flags=re.I) 
            for input_sentence in input_sentences]
        
def chatbot(input):
    pd.set_option('max_colwidth', 100)  
    data = pd.read_excel("WHO_FAQ.xlsx")
    module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

    # Create response embeddings
    response_encodings = module.signatures['response_encoder'](
        input=tf.constant(preprocess_sentences(data.Answer)),
        context=tf.constant(preprocess_sentences(data.Context)))['outputs']

# Create encodings for test questions
    question_encodings = module.signatures['question_encoder'](
        tf.constant(preprocess_sentences(input)))['outputs']

# Get the responses
    test_responses = data.Answer[np.argmax(np.inner(question_encodings, response_encodings), axis=1)]

    return " ".join(test_responses)

    
def chatbotresponse(input_msg):
    input_list = input_msg.split(' ')
    res = chatbot(input_list)
    return res
    
    
app = Flask(__name__)
app.static_folder = 'static'

@app.route('/', methods =["GET", "POST"])
def home():
    answer = "Hi, User. How may I assist you?"
    if request.method == "POST":
       # getting input with name = fname in HTML form
       question = request.form.get("question")
       answer = chatbotresponse(question) 
    return render_template('index.html',answer = answer)

if __name__ == '__main__':
   app.run(debug=True)