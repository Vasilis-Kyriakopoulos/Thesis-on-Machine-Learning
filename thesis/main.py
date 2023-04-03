import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from preprocessing import text_transform
import matplotlib.ticker as t
from sklearn import neighbors,metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron 
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import RidgeClassifier   
from sklearn.pipeline import make_pipeline
import sys

d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
np.set_printoptions(threshold=sys.maxsize)
data1 = pd.read_csv('datafiles/readydata1nostem.csv')
data2 = pd.read_csv('datafiles/readydata1stem.csv')
data3 = pd.read_csv('datafiles/readydata2nostem.csv')
data4 = pd.read_csv('datafiles/readydata2stem.csv')
label_mapping = {
    'fake':0,
    'true':1,
    'real':1
}
algorithms= [   
                   LogisticRegression(solver='saga',penalty='elasticnet',l1_ratio=0 ,class_weight='balanced'),
                   Perceptron(penalty="elasticnet",l1_ratio=1),
                   RidgeClassifier(solver='sag',tol=1e-3),
                   svm.SVC(kernel="sigmoid",cache_size=350  ),
                   KNeighborsClassifier(n_neighbors=14,weights='distance'),
                   BernoulliNB(alpha=0.1),
                   MultinomialNB(alpha=0.3),
                   tree.DecisionTreeClassifier(min_samples_split=0.5),
                   RandomForestClassifier() ,
                   MLPClassifier(hidden_layer_sizes=(10,)),  
                ]
names = [          
                    "count N-gram 1\nNo_Stem","tf-idf N-gram 1\nNo_Stem",
                    "count N-gram 2\nNo_Stem","tf-idf N-gram 2\nNo_Stem",
                    "count N-gram 1-2\nNo_Stem","tf-idf N-gram 1-2\nNo_Stem",
                    "count N-gram 1\nStem","tf-idf N-gram 1\nStem",
                    "count N-gram 2\nStem","tf-idf N-gram 2\nStem",
                    "count N-gram 1-2\nStem","tf-idf N-gram 1-2\nStem"          
        ]            

for algo in algorithms:
    accuracies_scores = []
    f1_scores = []
    precission_scores = []
    recall_scores=[]
    confusion_matrixs=[]
    for data in [data1,data2]:
        x = data['content'].values
        y = data['spam'].map(label_mapping).values
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1,stratify=y)
        models = []
        
        models.append(make_pipeline(CountVectorizer(ngram_range=[1,1]),algo))
        models.append(make_pipeline(TfidfVectorizer(ngram_range=[1,1]),algo))
        models.append(make_pipeline(CountVectorizer(ngram_range=[2,2]),algo))
        models.append(make_pipeline(TfidfVectorizer(ngram_range=[2,2]),algo))
        models.append(make_pipeline(CountVectorizer(ngram_range=[1,2]),algo))
        models.append(make_pipeline(TfidfVectorizer(ngram_range=[1,2]),algo))
        
        for model in models:
            #train model
            model.fit(x_train,y_train)
            prediction  = model.predict(x_test)
            accuracies_scores.append(round(metrics.accuracy_score(y_test,prediction),4))
            f1_scores.append(round(metrics.f1_score(y_test,prediction),4))
            confusion_matrixs.append(metrics.confusion_matrix(y_test,prediction))
    lis = ["count,N-gram 1,No Stem","tf-idf,N-gram 1,No Stem",
           "count,N-gram 2,No Stem","tf-idf,N-gram 2,No Stem",
           "count,N-gram 1-2,No Stem","tf-idf,N-gram 1-2,No Stem",
           "count,N-gram 1,Stem","tf-idf,N-gram 1,Stem",
           "count,N-gram 2,Stem","tf-idf,N-gram 2,Stem",
           "count,N-gram 1-2,Stem","tf-idf,N-gram 1-2,Stem"]
    with  open('results/'+type(algo).__name__[0:5].lower()+'_1.csv', 'w') as f:
        f.write("Vectorizer"+","+"N-gram"+","+"Stemming"+","+"Accuracy"+","+"F1 score\n")
        for i ,j,z in zip(accuracies_scores,f1_scores,lis):
            f.write(str(z))
            f.write(",")
            f.write(str(i))
            f.write(",")
            f.write(str(j))
            f.write("\n")
        f.close()
    ind = np.arange(12)
    width = 0.10
    plt.figure(figsize=(13,9))
    axes = plt.axes()
    plt.style.use('ggplot')
    plt.bar(ind/2, accuracies_scores, width,color='salmon',label ='Accuracy')
    plt.bar(ind/2 + width , f1_scores,width, color='lightseagreen',label = 'F1 score')
    plt.title(type(algo).__name__)
    plt.ylabel("Scores")
    plt.xticks(ind/2+width/2,names)
    plt.legend(loc='best')
    plt.setp(axes.get_xticklabels(),rotation=30, horizontalalignment='center')
    
    
    plt.savefig('eikones/d'+type(algo).__name__.lower()[0:5]+"_1")
    plt.figure(figsize=(13,7))
    for index,i in enumerate(confusion_matrixs):
        plt.subplot(3, 4, index+1)
        ax = sns.heatmap(i, annot=True,cmap='Blues',fmt='g')
        if index ==0 or index ==4 or index == 8:
            ax.set_ylabel('Actual Values ');
        ax.set_xlabel('\nPredicted Values')
        
        
        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])
    plt.savefig('eikones/f'+type(algo).__name__.lower()[0:5]+"_1")
    

    