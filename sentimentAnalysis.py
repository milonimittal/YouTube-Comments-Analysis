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
lemmatizer = WordNetLemmatizer() 

###Pre-processing begins###
def analyzingSentiments(dataset):
  #converting to lower case and removing all punctuations
  dataset.Comment = dataset.Comment.str.lower()
  dataset.Comment = dataset.Comment.str.replace('\n','').str.replace('[\'!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')

  #Combining all comments into one entity
  z = ''
  for i in range(len(dataset)):
      z = z+" "+dataset.iloc[i,:].Comment

  #Removing all emojis
  emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  
                             u"\U0001F300-\U0001F5FF"  
                             u"\U0001F680-\U0001F6FF"  
                             u"\U0001F1E0-\U0001F1FF"  
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"  
                             "]+", flags = re.UNICODE)
  z = emoji_pattern.sub(r'', z)

  unigrams = nltk.word_tokenize(z)

  #Removing all stopwords and including only those words which are in the english dictionary
  stop_words = stopwords.words('english')
  unigrams = [word for word in unigrams if word not in stop_words]
  setofwords = set(words.words())
  unigrams = [word for word in unigrams if (word in setofwords and len(word)>2 and word not in ["gon"])]
  count = Counter(unigrams) 

  ###Finding general opinion about channel###

  #extracting the top 1/4th words in terms of frequency
  most_common_element = count.most_common((int)(len(count)/4))

  # word=[]
  # num=[]
  # for element in most_common_element:
  #     word.append(element[0])
  #     num.append(element[1])

  # word_plot=word[:25]
  # num_plot=num[:25]
  # sns.set(rc={'figure.figsize':(13,8)})
  # g = sns.barplot(x=word_plot, y=num_plot)
  # g.set_xticklabels(g.get_xticklabels(), rotation=30)

  analyser = SentimentIntensityAnalyzer()
  pos = []
  neg = []

  for el in unigrams:
      score = analyser.polarity_scores(el)
      if ((score['compound']>0.3 ) ):
          pos.append(el)
      elif (( score['compound']<-0.3) ):
          neg.append(el)

  total=len(pos)+len(neg)
  channel_opinion = []

  #printing opinion words in the ratio of number of postive and negative words
  print("What do people think about this channel:")
  channel_opinion = channel_opinion+random.sample(pos,(int)(len(pos)*10.0/total))+random.sample(neg,(int)(len(neg)*10.0/total))
  for i in channel_opinion:
      print(i)

  return unigrams

  #most_common_element2 = count2.most_common(25)

  # word2=[]
  # num2=[]
  # for element2 in most_common_element2:
  #     word2.append(element2[0][0]+" "+element2[0][1])
  #     num2.append(element2[1])
  # sns.set(rc={'figure.figsize':(13,8)})
  # g = sns.barplot(x=word2, y=num2)
  # g.set_xticklabels(g.get_xticklabels(), rotation=30)