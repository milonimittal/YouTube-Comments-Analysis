import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage 
from tqdm import tqdm
import pandas as pd

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

def fetchingData(channel_title):
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()

    # channel_title = input("Enter the channel name: ")
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
    return dataset