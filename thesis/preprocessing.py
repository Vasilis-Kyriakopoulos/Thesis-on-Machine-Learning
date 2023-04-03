import csv
from datetime import time
import emoji
import re
from greek_stemmer import GreekStemmer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import time
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords

import unicodedata as ud


ss = SnowballStemmer(language="english")
grstemer = GreekStemmer()
translator = str.maketrans(string.punctuation,' '*len(string.punctuation))
en_stops = stopwords.words('english')

el_stops = stopwords.words('greek')    
for i in range(len(el_stops)):
    el_stops[i] = ''.join((c for c in ud.normalize('NFD', el_stops[i]) if ud.category(c) != 'Mn'))

en_stops = set(en_stops)
el_stops = set(el_stops)
stops  = en_stops.union(el_stops)

def strip_emoji(text):
    new_text = re.sub(emoji.get_emoji_regexp(), r" ", text)
    return new_text


def text_transform(s,en_stem,gr_stem):
    s  = s.lower()
    #remove urls
    s = re.sub(r'http\S+', '', s)
    #remove accents
    s = ''.join((c for c in ud.normalize('NFD', s) if ud.category(c) != 'Mn'))
    #remove punctuation marks
    s = re.sub(r'[^\w\s]',' ',s)
    #remove emojis
    s = strip_emoji(s)
    #remove \n and • 
    s = s.replace('\n',' ')
    s = s.replace('•',' ')
    #remove digits
    s = ''.join(i for i in s if not i.isdigit())
    words = s.split()
    
    for i in range(len(words)):
        if words[i]  in stops :
                words[i]=' '  
        if en_stem:
            words[i] = ss.stem(words[i])
        if gr_stem:
            words[i] =grstemer.stem(words[i].upper()).lower() 
             
    s = " ".join(words)
    return ' '.join(s.split())

if __name__ == "__main__":
    
    with open('datafiles/data1.csv', 'r',encoding='utf-8') as infile, open('datafiles/readydata1nostem.csv', 'w',encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        for row in reader:
            first = text_transform(row[0],False,False)
            second = row[10].lower()
            row_ = first+","+second+"\n"
            outfile.write(str(row_))
    with open('datafiles/data1.csv', 'r',encoding='utf-8') as infile, open('datafiles/readydata1stem.csv', 'w',encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        for row in reader:
            first = text_transform(row[0],True,True)
            second = row[10].lower()
            row_ = first+","+second+"\n"
            outfile.write(str(row_))
    

    with open('datafiles/data2.csv', 'r',encoding='utf-8') as infile, open('datafiles/readydata2nostem.csv', 'w',encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        for row in reader:
            first = text_transform(row[1],False,False)
            second = row[2]
            row_ = first+","+second+"\n"
            outfile.write(str(row_))

    with open('datafiles/data2.csv', 'r',encoding='utf-8') as infile, open('datafiles/readydata2stem.csv', 'w',encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        for row in reader:
            first = text_transform(row[1],True,False)
            second = row[2]
            row_ = first+","+second+"\n"
            outfile.write(str(row_))