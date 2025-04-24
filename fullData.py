# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import nltk
import re
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.model_selection import GridSearchCV


# Load the CSV files
df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

## concating title and text column
df_fake['text']=df_fake['title']+' '+df_fake['text']
df_true['text']=df_true['title']+' '+df_true['text']

# Display the first few rows of each dataset
print("Fake News Dataset:\n", df_fake.head())
print("\nTrue News Dataset:\n", df_true.head())

# Optional: Check basic information about the datasets
print("\nFake News Dataset Info:")
print(df_fake.info())
print("\nTrue News Dataset Info:")
print(df_true.info())

## creating a column 'class' to identify fake or True
df_fake['class']= 0 ## 0- fake
df_true['class']=1 ## 1-true

fake_train=df_fake.head(23472)
fake_test=df_fake.tail(10)

true_train=df_true.head(21418)
true_test=df_true.tail(10)

## concating train data sets fake and True
df_train=pd.concat([fake_train,true_train])

## concating test datasets fake and true
df_test=pd.concat([fake_test,true_test])

## shuffling the data in both data sets
df_train=df_train.sample(frac=1)
df_test=df_test.sample(frac=1)

## resetting the index
df_train=df_train.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)

## removing unneccessary columns
df_train=df_train[['text','class']]
df_test=df_test[['text','class']]

## checking for null 
df_train.isnull().sum()
df_test.isnull().sum()

stop_words = set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

## function to clean the data
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


df_train['text'] = df_train['text'].fillna('')  # Replace NaN with an empty string

## applying the clean function on the text column
df_train['text']=df_train['text'].apply(clean_text)



## splitting train test
from sklearn.model_selection import train_test_split
features=df_train.drop(columns='class')
target=df_train['class']

x_train,x_test,y_train,y_test=train_test_split(features,target,train_size=0.75,random_state=42)

## vectorizing
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(ngram_range=(1,2),max_df=0.8, min_df=5,max_features=10000, stop_words='english')
xv_train=vectorizer.fit_transform(x_train['text'])
xv_test=vectorizer.transform(x_test['text'])

## build the model
## Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(class_weight='balanced',criterion='gini',max_depth=10, min_samples_leaf=1, min_samples_split=   5)
dt.fit(xv_train,y_train)

## predict the test data 
dt_pred=dt.predict(xv_test)

##evaluation score for Decision Tree
from sklearn.metrics import accuracy_score, classification_report
print(f'score of DecisionTree: {dt.score(xv_test,y_test)}')

print('classification_report of dt',classification_report(y_test,dt_pred))
print('accuracy_score  of dt:',accuracy_score(y_test,dt_pred))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt, xv_train, y_train, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean CV Accuracy:", scores.mean())



## Random Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=0,class_weight='balanced')
rf.fit(xv_train,y_train)

## predict the test data
rf_pred=rf.predict(xv_test)

##evaluation score for Random forest
print(f'score of Random forest: {rf.score(xv_test,y_test)}')

print('classification_report of RF',classification_report(y_test,rf_pred))
print('accuracy_score of RF:',accuracy_score(y_test,rf_pred))

##Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(xv_train,y_train)

##predict
nb_pred=nb.predict(xv_test)

##evaluation score for Random forest
print(f'score of Naive bayes: {nb.score(xv_test,y_test)}')

print('classification_report of NB',classification_report(y_test,nb_pred))
print('accuracy_score of NB:',accuracy_score(y_test,nb_pred))


## predicting with test data set
print(df_test.shape)

features=df_test['text']
target=df_test['class']

features=features.apply(clean_text)
xv_test=vectorizer.transform(features)

test_pred=dt.predict(xv_test)

print('accuracy score :',accuracy_score(target,test_pred))


## model evaluation
def out_label(n):
    return 'Fake News'if n ==0 else 'True News'
    

def manual_testing(news):
    testing_news={'text':[news]}
    df_news=pd.DataFrame(testing_news)
    feature=df_news['text']
    feature=feature.apply(clean_text)
    xv_test=vectorizer.transform(feature)
    dt_predi=dt.predict(xv_test)
    rf_predi=rf.predict(xv_test)
    nb_predi=nb.predict(xv_test)
    return print('\n \n dt predictions :{} \n rf predictions :{} \n nb prediction :{}'.format(out_label(dt_predi[0]),out_label(rf_predi[0]),out_label(nb_predi[0])))

news='''"SAY GOOD BYE TO LONDON: Radical Muslim WINS London’s Mayoral Election By Over 300,000 Votes","Has the entire world gone mad with political correctness? Is there any chance of saving the UK from itself? UPDATE: Labour Party politician Sadiq Khan has been elected London mayor   the first Muslim to lead Europe s largest city.Election officials say Khan defeated Conservative rival Zac Goldsmith by more than 300,000 votes, after first- and second-preference votes were allocated.The result came early Saturday, more than 24 hours after polls closed.Khan was elected to replace Conservative Mayor Boris Johnson after a campaign marked by U.S.-style negative campaigning.Goldsmith, a wealthy environmentalist, called Khan divisive and accused him of sharing platforms with Islamic extremists.Khan, who calls himself  the British Muslim who will take the fight to the extremists,  accused Goldsmith of trying to scare and divide voters in a proudly multicultural city of 8.6 million people   more than 1 million of them Muslim.Londoners go to the polls on Thursday to choose a replacement for the outgoing Mayor, Boris Johnson MP. Despite the fact the Labour Party is currently mired in an anti-Semitism scandal, if all things remain equal, expect the party s candidate Sadiq Khan MP to be confirmed in the early hours of Friday morning.Mr. Khan, 45, has had a successful career in the Labour Party, being elected to parliament in 2005, becoming a Minister of State in 2008 with a promotion in 2009. He was a Shadow Secretary of State for Justice from 2010-15, and has been running for London Mayor since then.POLLINGIn fact, as I predicted in January, the polls have changed very little since the beginning of the year. This is despite a negative-ad onslaught by the Conservative Party and its candidate Zac Goldsmith   the multi-millionaire son of Eurosceptic royalty Sir James Goldsmith, and brother of socialite and Vanity Fair editor Jemima Khan.Mr. Goldsmith and his  Back Zac  campaign have used the last few months to highlight Sadiq Khan s proximity to Islamic extremists, extremism, and this past weekend, to anti-Semitism.But perhaps it speaks to the mindset of Londoners, and certainly the British capital s demographic shift, that such news has scarcely affected Mr. Khan.In January, a YouGov poll put Mr. Goldsmith on 35 to Mr. Khan s 45 per cent. When you take into account London s supplementary voting system, the numbers after the second preference votes are counted ended up 45-55 to Mr. Khan.Last week, that number stood at 40-6o to Mr. Khan. After second preference votes, the extremist-adjacent candidate has a 20 point poll lead.While last week s events   when one of Mr. Khan s most prominent backers Ken Livingstone was implicated in a Hitler/anti-Semitism scandal   may serve to keep some of Mr. Khan s voters at home, it is hard to imagine the Conservatives overturning such a drastic poll lead.A MUSLIM MAYOR?Polling suggests some people are nervous about having someone like Mr. Khan near an office that wields so much power, responsibility, and cash.Private conversations with Westminster insiders often see Lutfur Rahman   the former Mayor of Tower Hamlets   raised as another example of a prominent Muslim mayor.Mr. Rahman was removed from office, accused by critics of playing sectarian politics with the area s Muslim population, of backing Islamists, and of distributing tax payer cash to his favoured Muslim groups to secure their support.Mr. Rahman was found guilty of  corrupt and illegal practices    and has perhaps set back the plight of the few, integrated British Muslims in elected life. He   alongside politicians like Humza Yousaf, Sayeeda Warsi, Rushanara Ali, Shabana Mahmood, Yasmin Qureshi, Amjad Bashir, Naz Shah, and Tasmina Ahmed-Sheikh   have created a deep distrust between British voters and Muslim politicians.In fact one third of Londoners remain suspicious of having a Muslim Mayor, and the likes of Sajid Javid or Syed Kamall suffer because of their co-religionists  insistence on fellow-travelling with extremists, if not holding extremist views themselves.EXTREMISMAnd Mr. Khan can hardly claim a clean record. Mr. Goldsmith s attacks are not without basis, though they have been shrugged off as  racism  or  Islamophobia  with the assistance of the left s useful idiots like Owen Jones.Apart from his somewhat threatening statements about not voting for him while claiming that  he is the West , Mr. Khan s own track record is perhaps one of the most sour of all Muslim politicians in the Western world.In 2001 he was the lawyer for the Nation of Islam in its successful High Court bid to overturn the 15-year-ban on its leader, Louis Farrakhan.In 2005 and 2006 he visited terror-charged Babar Ahmad in Woodhill Prison. Mr. Ahmed was extradited to the U.S. in 2012, serving time in prison before being returned to the UK in 2015. Mr. Ahmed pleaded guilty to the terrorist offences of conspiracy, and providing material support to the Taliban.And Mr. Khan also campaigned for the release and repatriation of Shaker Aamer, Britain s last Guantanamo detainee, who was returned to the UK in November.Both Messrs Aamer and Ahmed provided Mr. Khan with links to the advocacy group CAGE, which described the Islamic State executioner Mohammed Emwazi as a  beautiful young man , and which has campaigned on behalf of both men. Mr. Khan is reported to have shared a stage with five Islamic extremists, including at sex-segregated events. Even so, his poll numbers remain firm.On Friday morning, Londoners will likely get the news that their mayor for the next four years is a man with the judgement, priorities, and fellow travellers as laid out above. This, combined with an annual  16bn budget, and an army of police, bureaucrats, and officials, would make Mr. Khan one of the most powerful Muslims in the Western world.For entire story: Breitbart News ",left-news,"May 3, 2016"'''

manual_testing(news)



import joblib
import pickle
import os

os.makedirs('models',exist_ok=True)

models={"Decision Tree": dt,
    "Random Forest": rf,
    "Naive Bayes": nb
    }

joblib.dump(models,'models/models.pkl')
joblib.dump(vectorizer,'models/vectorizer.pkl')
