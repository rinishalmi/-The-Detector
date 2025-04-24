import joblib
import streamlit as st
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

models=joblib.load("models/models.pkl")
vectorizer=joblib.load('models/vectorizer.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

def clean_text(text):
    text=text.lower()   ## converts into lowercase
    text=re.sub(r'[^a-zA-Z0-9\s]', '', text)  ## remove speacial characters
    text=re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text=re.sub(r"http\S+", "", text)  # remove URLs
    text=text.strip() ## remove trailing/leading white spaces
    text=word_tokenize(text)
    text=[lemmatizer.lemmatize(word) for word in text]
    text=' '.join([word for word in text if word not in stop_words])
    return text

from collections import Counter

def predicting_news(news):
    cleaned_news=clean_text(news)
    vectorized=vectorizer.transform([cleaned_news])
    votes=[]

    for model in models.values():
       pred=model.predict(vectorized)
       votes.append(pred[0])

    from collections import Counter
    votes_counts=Counter(votes)
    final_votes=votes_counts.most_common(1)[0][0]
    return "True News ✅" if final_votes == 1 else "Fake News ❌"

## streamlit UI

st.set_page_config(page_title='Fake News Detector',page_icon='📰')
st.title('Fake News detector 📰')
st.write('Enter a news article to check if its Fake or True')


user_input=st.text_area('paste your news article here.')

if st.button('check news'):
    if user_input.strip():
        result=predicting_news(user_input)
        if result.startswith('Fake'):
            st.error(result)
        else:
            st.success(result)
    else:
        st.warning('please enter some news content')


