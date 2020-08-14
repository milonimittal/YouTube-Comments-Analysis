import sentimentAnalysis
import youtubeAPI
import topicsOfDiscussion

class YoutuberAnalysis:
	def __init__(self, channel_title):
		self.channel_title=channel_title
		self.dataset=youtubeAPI.fetchingData(self.channel_title)
		self.unigrams=sentimentAnalysis.analyzingSentiments(self.dataset)
		topicsOfDiscussion.findingTopics(self.unigrams)