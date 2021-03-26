import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

#df es el archivo csv donde estan los datos para entrenar el modelo
# subir archivo csv con el 70% de los datos
df_train = pd.read_csv('data_model_evaluation/corpus.csv').dropna(how='all')
#test es el archivo csv donde estan los datos para probar el modelo
#f_test = pd.read_csv('data_model_evaluation/corpus_test.csv').dropna(how='all')#subir archivo csv con el 30% de los datos


#data cleaning

#drop a specific column (Unnamed: 2) from the corpus.csv
#df_train.drop('Unnamed: 2',axis='columns', inplace=True)

#drop a specific column (Unnamed: 3) from the corpus.csv
#df_train.drop('Unnamed: 3',axis='columns', inplace=True)

#lowercase for the data in the corpus

df_train['tuits'] = df_train['tuits'].str.lower()
df_train['sentiment'] = df_train['sentiment'].str.lower()



#se quita las polaridades neutra del df-train y se combierte las polaridades a variable binaria 1-positivo 0-negativo
df_train['polarity'] = 0
df_train.polarity[df_train.sentiment.isin(['positivo'])] = 1
df_train.polarity[df_train.sentiment.isin(['violento', 'negativo'])] = 0
df_train.polarity.value_counts(normalize=True)
print(df_train.head())
#delete new polarity
df_train.drop('sentiment', axis='columns', inplace=True)

#partición del corpus
train, test = train_test_split(df_train, test_size=0.2, random_state=1)
X_train = train['tuits'].values
X_test = test['tuits'].values
y_train = train['polarity']
y_test = test['polarity']

#limpieza y pre-procesamiento


def tokenize(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)


def stem(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

en_stopwords = set(stopwords.words("spanish"))

#extraccio4n de caracteristicas
vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    ngram_range=(1, 1),
    stop_words=en_stopwords
)

#modelado
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
np.random.seed(1)

pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1
                    ) 
grid_svm.fit(X_train, y_train)
print(grid_svm.score(X_test, y_test))

