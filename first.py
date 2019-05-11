# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Dataset 
corpus = []
dataset = pd.read_csv('final_dataset.tsv',delimiter='\t',quoting = 3, names = ['id','summary','name','genre'],nrows=8000)
const=4000

# Stemming words and removing stopwords
y_corpus = []
for i in range(const):
    review = re.sub('[^a-zA-Z]', ' ', dataset['summary'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    y_review = dataset['genre'][i]
    y_review = re.sub('[^0-9]', ' ', y_review)
    y_review = y_review.split()
    y_review = ','.join(y_review)
    #print(y_review)
    y_corpus.append(y_review)    
y = []
for i in y_corpus:
    li = [0 for _ in range(6)]
    for j in i:
        if(j==','):
            continue
        else:
            #print(j,type(j))
            li[int(j)] = 1
    #print(li)
    y.append(li)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100000)
X = cv.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB

# with a gaussian naive bayes classifier
#Naive bayes gives good result because it assumes features are independent just like the words in the summary are.
classifier = GaussianNB()
tr=int(0.8*const)
for i in range(6):
    y_int=[]
    for j in range(tr):
        y_int.append(y_train[j][i])
    classifier.fit(X_train, y_int)
    predictions = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    y_dash=[]
    for j in range(const-tr):
        y_dash.append(y_test[j][i])
    mat=confusion_matrix(predictions,y_dash)
    print(mat)
#Overall Accuracy = 70.0833% average Least for drama and comedy highest for horror

# with a Linear SVM classifier
# Linear SVM is good for high dimensionality problems like NLP
tr = int(0.8*const)
from sklearn.svm import SVC    
clfier = SVC(kernel='linear',probability=True)
from sklearn.metrics import confusion_matrix
for i in range(6):
    y_int=[]
    for j in range(tr):
        y_int.append(y_train[j][i])
    clfier.fit(X_train, y_int)
    predictions = clfier.predict(X_test)
    y_dash=[]
    for j in range(const-tr):
        y_dash.append(y_test[j][i])
    mat=confusion_matrix(predictions,y_dash)
    print(mat)
#Overall accuracy = 68.50%



#Pre processing code - Merging files and data cleaning
'''summary_dataset = pd.read_csv('plot_summaries.tsv', delimiter = '\t', quoting = 3, names = ['id','summary'])
genre_dataset = pd.read_csv('genre_data.tsv', delimiter = '\t', quoting = 3, names = ['id','name','genre'])

import ast
genre_dataset['genre']=genre_dataset['genre'].apply(ast.literal_eval).apply(lambda x:list(x.values()))
dataset = pd.merge(summary_dataset,genre_dataset,on="id")
#dataset.to_csv('out_small.tsv',sep='\t',encoding='utf-8')

corr_data = []
d = list(dataset['genre'])
c=0
for i in range(len(d)):
    temp=[]
    temp.append(i)
    for j in d[i]:
        if(j in ['Comedy','Action','Drama','Thriller','Horror','Fiction','Romance Film']):
            temp.append(j)
    if(len(temp)!=1):
        corr_data.append(temp)
c=0
final_dataset=[]
for i in range(len(dataset)):
    if(i==corr_data[c][0]):
        temp=list(dataset.iloc[c,:-1].values)
        temp.append(corr_data[c][1:])
        final_dataset.append(temp)
        c=c+1
r = pd.DataFrame(final_dataset)
r.to_csv('final_dataset.tsv',sep='\t',encoding='utf-8')
'''