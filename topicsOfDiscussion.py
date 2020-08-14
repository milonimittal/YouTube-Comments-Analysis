import pandas
import nltk
import itertools
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from nltk.corpus import words
from nltk import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def findingTopics(unigrams):
    count = Counter(unigrams) 
    bigrams = ngrams(unigrams, 2)
    count2 = Counter(bigrams)
    num = len(list(ngrams(unigrams, 2)))
    arr=[];
    for i in count2:
        o11=count2[i]
        o12=count[i[0]]-count2[i]
        o21=count[i[1]]-count2[i]
        o22=num-o11-o12-o21
        chi=(num*(o11*o22-o12*o21)*(o11*o22-o12*o21))/(((o11+o12)*(o11+o21)*(o12+o22)*(o21+o22)))
        if (o12+o21>30 and chi>3.841): 
            arr.append((i,chi))

    arr.sort(key=takeSecond, reverse=True)

    #Tagging the bigrams
    arr_final=[]
    for i in arr:
        arr_final.append(nltk.pos_tag(i[0]))

    #Extracting words of the form <noun,noun> or <adjective,noun>
    ct=0
    channel_about=[]
    for el in arr_final:
        if(ct<=10):
            if(el[0][1]=='NN' and el[1][1]=='NN'):
                channel_about.append(el[0][0]+" "+el[1][0])
                ct+=1
            elif (el[0][1]=='JJ' and el[1][1]=='NN'):
                channel_about.append(el[0][0]+" "+el[1][0])
                ct+=1    

    #Printing top 10 thngs that people talk about in the comments section
    print("Other things people talk about: ")
    for i in channel_about:
        print(i)

    #Creating a wordcloud of all frequently used words
    stopwords = set(STOPWORDS)
    comment_words=''
    for words in unigrams: 
        comment_words = comment_words + words + ' '
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 
        ).generate(str(comment_words))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()


def takeSecond(elem):
    return elem[1]