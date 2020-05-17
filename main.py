################################################################################################################################
                                                    #Extracting Data Using APIs#
################################################################################################################################
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage 


CLIENT_SECRETS_FILE = "./client_secret.json" 


SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service(): 
    credential_path = os.path.join('./', 'credential_sample.json')
    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRETS_FILE, SCOPES)
        credentials = tools.run_flow(flow, store)
    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials) 

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
service = get_authenticated_service()

channel_title = input("Enter the channel name: ")
query_results = service.search().list(
    part = 'snippet',
    q = channel_title,
    order = 'viewCount',
    type = 'channel',
    ).execute()

channelID = query_results['items'][0]['id']['channelId']

request = service.channels().list(
    part = "statistics",
    id = channelID
    ).execute()

subCount = request['items'][0]['statistics']['subscriberCount']
print("Subscriber Count: "+subCount)

query_results = service.search().list(
    part = 'snippet',
    order = 'date', 
    maxResults = 10,
    type = 'video', 
    relevanceLanguage = 'en',
    safeSearch = 'moderate',
    channelId = channelID
    ).execute()

video_id = []
channel = []
video_title = []
video_desc = []
for item in query_results['items']:
    video_id.append(item['id']['videoId'])
    channel.append(item['snippet']['channelTitle'])
    video_title.append(item['snippet']['title'])
    video_desc.append(item['snippet']['description'])

video_id_pop = []
channel_pop = []
video_title_pop = []
video_desc_pop = []
comments_pop = []
comment_id_pop = []
reply_count_pop = []
like_count_pop = []

from tqdm import tqdm
for i, video in enumerate(tqdm(video_id, ncols = 100)):
    response = service.commentThreads().list(
                    part = 'snippet',
                    videoId = video,
                    maxResults = 100, 
                    order = 'relevance', 
                    textFormat = 'plainText',
                    ).execute()
    
    comments_temp = []
    comment_id_temp = []
    reply_count_temp = []
    like_count_temp = []
    for item in response['items']:
        comments_temp.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        comment_id_temp.append(item['snippet']['topLevelComment']['id'])
        reply_count_temp.append(item['snippet']['totalReplyCount'])
        like_count_temp.append(item['snippet']['topLevelComment']['snippet']['likeCount'])
    comments_pop.extend(comments_temp)
    comment_id_pop.extend(comment_id_temp)
    reply_count_pop.extend(reply_count_temp)
    like_count_pop.extend(like_count_temp)
    
    video_id_pop.extend([video_id[i]]*len(comments_temp))
    channel_pop.extend([channel[i]]*len(comments_temp))
    video_title_pop.extend([video_title[i]]*len(comments_temp))
    video_desc_pop.extend([video_desc[i]]*len(comments_temp))
    
query_pop = [channel_title] * len(video_id_pop)

import pandas as pd

output_dict = {
        'Query': query_pop,
        'Channel': channel_pop,
        'Video Title': video_title_pop,
        'Video Description': video_desc_pop,
        'Video ID': video_id_pop,
        'Comment': comments_pop,
        'Comment ID': comment_id_pop,
        'Replies': reply_count_pop,
        'Likes': like_count_pop,
        }

dataset = pd.DataFrame(output_dict, columns = output_dict.keys())



################################################################################################################################
                                                    #Data Processing#
################################################################################################################################

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

###Pre-processing ends###

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


bigrams = ngrams(unigrams, 2)
count2 = Counter(bigrams)








#most_common_element2 = count2.most_common(25)

# word2=[]
# num2=[]
# for element2 in most_common_element2:
#     word2.append(element2[0][0]+" "+element2[0][1])
#     num2.append(element2[1])
# sns.set(rc={'figure.figsize':(13,8)})
# g = sns.barplot(x=word2, y=num2)
# g.set_xticklabels(g.get_xticklabels(), rotation=30)







###Finding other things that people talk about in the comments section###

#Applying chi-square testing to find useful collocations
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
def takeSecond(elem):
    return elem[1]
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





