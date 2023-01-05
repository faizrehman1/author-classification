





# initialize afinn sentiment analyzer

from afinn import Afinn
af = Afinn()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os,string#,path

#wordCloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from urllib.request import urlopen
from PIL import Image

#Text Processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# %matplotlib inline

#ML Model
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgbm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

#Optimisation
import pickle
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING) #Disable Warnings
import nltk
nltk.download('stopwords')
nltk.download('punkt')

"""
#FUNCTIONS
"""
def preprocess_sentence(df): #returns the whole sentence, with preprocessed text
    word_list = []
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', df) #remove punctuations
    #text = text.lower() #lower case
    tokenized_word=word_tokenize(text) #separate into words
    for word in tokenized_word:
        if word not in stop_words: #filter stop-words
            word = stem.stem(word) #stemming
            word_list.append(word) #append to general list
    return ' '.join(word_list) #rejoins the sentence without the stopwords

def process_list(text): #returns a list of preprocessed words
        word_list = []
        #for t in text:            
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) #remove punctuations
        text = text.lower() #lower case
        tokenized_word=word_tokenize(text) #separate into words
        for word in tokenized_word:
            if word not in stop_words: #filter stop-words
                word = stem.stem(word) #stemming
                word_list.append(word) #append to general list
        return word_list
    
def build_freqs(texts, author):
    authorslist = np.squeeze(author).tolist()
    # Start with an empty dictionary and populate it by looping over all samples
    # and over all processed words in each sample.
    freqs = {}
    words_sample = []
    for text, author in zip(texts,authorslist):
        for word in process_list(text):
            words_sample.append(word)
            pair = (word, author)
            freqs[pair] = freqs.get(pair, 0) + 1  
    return freqs,words_sample

"""# 2. NLP Model


"""

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

"""## 2.1 Data Preprocessing

For the Data Analysis section we played with preprocessing. Now we are going to prepare our dataset for our ML model:
* Encode the Labels
* Pre-Process text samples (stemming, stopwords, lowercase, remove punctuations)
* Oragnise the Training, Validation and Test Sets
* Use CountVectorizes to build the word vector

An additional step that has improved performance was to add the most common words to the **Stop_Words** list. The model improved by removing *one, could, and would* from the samples. In addition, not transforming the words to lowercase also increased model performance.

>Data Preprocessing and Creation of Training and Test Sets
"""
stop_words=set(stopwords.words("english"))
stem = PorterStemmer()
stop_words.update(('one','could','would'))

df_train = pd.read_csv('train-authors - train-authors.csv')
df_test = pd.read_csv('test-authors - test-authors.csv')

df_train['text_pre'] = df_train['text'].apply(lambda x : preprocess_sentence(x))
df_test['text_pre'] = df_test['text'].apply(lambda x : preprocess_sentence(x))

LabelEnc = preprocessing.LabelEncoder()
target_train = LabelEnc.fit_transform(df_train.author.values)

features_train = df_train.text_pre.values

features_test = df_test.text_pre.values

# Create First Train and Test sets
x_train, x_test, y_train, y_test = train_test_split(features_train, target_train, test_size=0.20,random_state=123)

print ("Training set size", x_train.shape[0])
print ("Test set size",x_test.shape[0])

""">Using CountVectorize to transform the data and train our Naive-Bayes and Logistic Regression Models"""

CVec = CountVectorizer(analyzer='word',ngram_range=(1, 3),dtype=np.float32)

# Fitting Count Vectorizer to training and test sets
x_train_CVec =  CVec.fit_transform(x_train) 
x_test_CVec = CVec.transform(x_test)

submission_test = CVec.transform(features_test)

"""The models used for this study are the classic Naive-Bayes (NB) and Logistic Regression (LR), commonly used for NLP tasks. In addition, I am also using LGBM as it usually provides a good trade-off between accuracy and training time.

The model hyperparameters were optimised using the Optuna Library. The code for the optimisation is commented out as it requires a long time for the LGBM and LR.
"""


def CV(model,x_train,x_test):
    logloss = []
       
    model.fit(x_train, y_train)
    predictions = 0
    logloss.append(0)
    test_score = 0
    mean_res = np.mean(logloss)
    std_dev = np.std(logloss)
    return mean_res,std_dev, test_score, model    



#LR_Model = CV(LR,x_train_CVec,x_test_CVec)
#NB_Model = CV(NB,x_train_CVec,x_test_CVec)
#LGBM_Model = CV_LGBM(clf_LGBM,x_train_CVec,x_test_CVec)

"""An additional ensemble model is created to verify if such strategy could provide any significant improvement to our model. The sklearn library provides the VotingClassifier module to facilitate the ensemble model construction."""

from sklearn.ensemble import VotingClassifier
LR = LogisticRegression(C=1, solver = 'lbfgs', max_iter = 1000)
NB = MultinomialNB(alpha = 1.3)
ensemble = VotingClassifier(estimators=[('LR', LR), ('NB', NB)], voting='soft', weights=[1,1.5])
Ensemble_Model = CV(ensemble,x_train_CVec,x_test_CVec)

"""# 3. Result

Here we discuss the results from the modelling strategy we applied. The print statement below show us the LogLoss for each model for the Cross-Validation and Test sets.
"""

#print("Logistic Regression \n LogLoss: %.3f +/- %.4f \n Test Set LogLoss: %.3f" % (LR_Model[0],LR_Model[1],LR_Model[2]))
#print("Naive-Bayes \n LogLoss: %.3f +/- %.4f \n Test Set LogLoss: %.4f" % (NB_Model[0],NB_Model[1],NB_Model[2]))
#print("LGBM \n LogLoss: %.3f +/- %.4f \n Test Set LogLoss: %.3f" % (LGBM_Model[0],LGBM_Model[1],LGBM_Model[2]))
#print("Ensemble NB and LR \n LogLoss: %.3f +/- %.4f \n Test Set LogLoss: %.3f" % (Ensemble_Model[0],Ensemble_Model[1],Ensemble_Model[2]))

#LR_preds = LR_Model[3].predict(x_test_CVec)
#NB_preds = NB_Model[3].predict(x_test_CVec)
#Ensemble_preds = Ensemble_Model[3].predict(x_test_CVec)

categories = [df_train['author'].unique()]


#CMatrix(LR_preds,'Logistic Regression')
#CMatrix(NB_preds,'Naive-Bayes')
#CMatrix(Ensemble_preds,'Ensemble')

ensemble_preds = Ensemble_Model[3].predict(submission_test)
ids = df_test.copy()
predict = pd.DataFrame(ensemble_preds, columns=['predicted_author_name'] )
predict['predicted_author_name'] = LabelEnc.inverse_transform(predict['predicted_author_name'].values)
submission = pd.concat([ids, predict] ,axis = 1)
submission.to_csv('results.csv',index=False)
submission.head()

from sklearn.metrics import f1_score
print("F1-Score is :",f1_score(submission['author'], submission['predicted_author_name'],average='weighted'))