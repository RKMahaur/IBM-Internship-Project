# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:02:31 2024

@author: 91981
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import pickle

# loading the saved PorterStemmer
ps = pickle.load(open('ps.pkl','rb'))

# loading the saved CountVectorizer
cv = pickle.load(open('cv.pkl','rb'))

# loading the saved Model
model = pickle.load(open('model.pkl','rb'))

# Function for preprocessing and sentiment prediction
def predict_sentiment(sample_review):
  # Text preprocessing
  sample_review = re.sub(pattern = '[^a-zA-Z]',repl=' ',string = sample_review)  # Remove special characters
  
  sample_review = sample_review.lower() # Lowercase the text
  
  sample_review_words = sample_review.split() # Tokenize
  
  # Removing stopwords and stemming
  sample_review_words = [ps.stem(word) for word in sample_review_words if not word in stopwords.words('english')]

  final_review = ' '.join(sample_review_words)
  
  # Vectorizing the input
  temp = cv.transform([final_review]).toarray()
  
  # Model prediction
  prediction = model.predict(temp)

  if prediction == 1:
    return('This is a POSITIVE review.')
  else:
    return('This is a NEGATIVE review.')

# Main function for the app
def main():
    
    # giving a title
    st.title('Sentiment Analysis on Restraunt Reviews')
    
    # getting the input data from the user
    user_review = st.text_input('Enter a review for the Restraunt:')
    
    # code for Prediction when the button is clicked
    if st.button('Get Sentiment Result'):
        if user_review:
            sentiment = predict_sentiment(user_review)
            st.success(sentiment)
        else:
            st.error("Please enter a review.")
        
# Running the app
if __name__ == '__main__':
    main()
