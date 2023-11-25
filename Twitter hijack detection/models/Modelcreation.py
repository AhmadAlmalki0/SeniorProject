import pandas as pd
import numpy as np

import tensorflow_hub as hub

import pandas as pd

import tensorflow_text as text

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

import numpy as np

df = pd.read_csv('/content/gdrive/MyDrive/DB_projects_private/text.csv')
df.head()

#Cleaning the text data
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy

import nltk
nltk.download('stopwords')

# Load spacy
nlp = spacy.load('en_core_web_sm')

def clean_string(text, stem="None"):

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("arabic")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string
    
clean_string('متوفره جميع خدمات تويتر ..\n⚠️دعم زيادة متابعي')

#let's convert all the tweets to string before transofrming 

df['Content.Text'] = df['Content.Text'].values.astype(str)

df['Content.Text'] = df['Content.Text'].apply(lambda x: clean_string(x))

df['target'].value_counts()

print(str(round(6328/22659,2))+'%')

df['target'].unique()[1]


# creating 2 new dataframe as df_ham , df_spam

df_true = df[df['target']==df['target'].unique()[1]]

df_false = df[df['target']==df['target'].unique()[0]]

print("true Dataset Shape:", df_true.shape)

print("false Dataset Shape:", df_false.shape)

# downsampling ham dataset - take only random 747 example
# will use df_spam.shape[0] - 747

df_true_downsampled = df_true.sample(df_false.shape[0])
df_true_downsampled.shape


# concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
df_balanced = pd.concat([df_false , df_true_downsampled])
df_balanced['target'].value_counts()

# Encoding the target variable
df_balanced['ad'] = df_balanced['target'].apply(lambda x:1 if x==df['target'].unique()[1] else 0)
df_balanced.head()

df_balanced['target'].unique()[0]

#splittign the dataset
# loading train test split
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(df_balanced['Content.Text'], df_balanced['target'],
                                                    stratify = df_balanced['ad'])
                                                    
    

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC



# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()),])


text_clf_lsvc.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_lsvc.predict(X_test)

import matplotlib.pyplot as plt
from sklearn import metrics
# Calculate the confusion matrix

conf_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=predictions)

# Print the confusion matrix using Matplotlib

fig, ax = plt.subplots(figsize=(6, 6))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()



# Print a classification report
print(metrics.classification_report(y_test,predictions))


# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


import joblib

joblib.dump(text_clf_lsvc,'pipe.joblib')

