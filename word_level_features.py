import json 
import numpy as np
import spacy
import string
from nltk.tree import Tree 
from stanfordcorenlp import StanfordCoreNLP
from textblob import TextBlob
nlp = spacy.load('en')
movie_kb = json.load(open('movie_kb.json','r'))
aspects = json.load(open('aspects.json','r'))

class WordFeatures(object):
	aspects_unigram = aspects[0]
	aspects_bigram = aspects[1]	
	ner_aspect_mapper = {'ACTOR': 'acting' ,'DIRECTOR': 'direction', 'PRODUCER' : 'production', 'CINEMATOGRAPHER': 'cinematography' , 'CHARACTER': 'character','MUSIC_DIRECTOR':'music'}
	aspect_keys = aspects_unigram.keys()

	def init(self):
		print(self.aspects_unigram)
		#self.sentiment_nlp = StanfordCoreNLP('/home/nikita/Downloads/stanford-corenlp-full-2018-02-27')
		#self.props = {'annotators':'sentiment','pipelineLanguage':'en','outputFormat':'json'}

		#nlp = spacy.load('en')
		#self.ner_aspect_mapper = {'ACTOR': 'acting' ,'DIRECTOR': 'direction', 'PRODUCER' : 'production', 'CINEMATOGRAPHER': 'cinematography' , 'CHARACTER': 'character','MUSIC_DIRECTOR':'music'}
		#movie_kb = json.load(open('movie_kb.json'))
		#print(len(movie_kb))
		#self.movie_kb = movie_kb
	
		#self.aspects_unigram = aspects[0]
		#self.aspects_bigram = aspects[1]
		print(self.aspects_bigram)
		self.aspect_keys = aspects_unigram.keys()
		aspect_mapper = {}
		for pos,key in enumerate(aspect_keys):
			aspect_mapper[key] = pos
		self.aspect_mapper = aspect_mapper

	def get_special_ner(self,word,curr_ner,imdb_id): #check if word is a special category word . Handcrafterd convert this to spacy code later
			kb = movie_kb[imdb_id]
			word = word.lower()
			if len(curr_ner) > 1:
				if word in kb['actor']:
					return "ACTOR"
				if word in kb['director']:
					return "DIRECTOR"
				if word in kb['cinematographer']:
					return "CINEMATOGRAPHER"
				if word in kb['musician']:
					return "MUSICDIRECTOR"
				if word in kb['writer']:
					return "WRITER"
				if word in kb['character']:
					return "CHARACTER"
				if word in kb['producer']:
					return "PRODUCER"
				
			return curr_ner

	def get_unigram_aspect(self,word,word_ner):
			word = word.lower()
			for key in self.aspects_unigram.keys():
				if word in self.aspects_unigram[key]:
					return key, 1
			
			if word_ner in self.ner_aspect_mapper:
				return self.ner_aspect_mapper[word_ner], 1
			
			return "none", 0

	def get_bigram_aspect(self,bigram): 
		for key in self.aspects_bigram.keys():
			if bigram.lower() in self.aspects_bigram[key]:
				return key,key,1,1

		return "none","none",0,0		

	def in_query(self,document_tokens,query_tokens):
		marker = np.zeros(len(document_tokens),dtype=int)
		query_tokens_l = [token.lower() for token in query_tokens]
		for j,token in enumerate(document_tokens):
			if token.lower() in query_tokens_l and token not in string.punctuation:
				marker[j] = 1
		return list(marker)

	def check_aspect_word(self,tokens,tokens_ner): #binary bit
		marker = np.zeros(len(tokens),dtype=int)
		aspect_list = ["none" for i in marker]
		for loc in range(len(tokens)-1):
			if marker[loc]==0:
				bigram_string = tokens[loc] + " " + tokens[loc+1]
				aspect_list[loc],aspect_list[loc+1],marker[loc],marker[loc+1] = self.get_bigram_aspect(bigram_string)
				
		for loc in range(len(tokens)):
			if marker[loc] == 0:
				aspect_list[loc], marker[loc] = self.get_unigram_aspect(tokens[loc], tokens_ner[loc]) 
					
		return aspect_list, list(marker)

	def check_domain_word(self,tokens): #WIP in terms of data
		marker = np.zeros(len(tokens),dtype=int)
		for j,token in enumerate(tokens):
			if token.lower() in domain_words:
				marker[j] = 1
		return list(marker)

	def check_genre_word(self,tokens,imdb_id): #WIP
		pass 

	def get_token_lemma_pos_ner_stop(self,doc,imdb_id="",extended_ner=True):
		
		doc = nlp(doc.decode('utf-8'))
		token_text = []
		lemma = []
		pos = []
		ner = []
		stop = []

		for token in doc:
			token_text.append(token.text)
			lemma.append(token.lemma_)
			pos.append(token.pos_)
			ner.append(token.ent_type_)
			stop.append(token.is_stop)	
	   
		if extended_ner:
			extended_ner = []
			for word,word_ner in zip(token_text,ner):
				extended_ner.append(self.get_special_ner(word,word_ner,imdb_id))
			ner = extended_ner
		new_ner = []
		for i in ner:
			if  len(i) > 1:
				new_ner.append(i)
			else:
				new_ner.append('NO_NER')	
		stop = list(np.array(stop) * 1)
		return token_text, lemma, pos, new_ner, stop

	def get_tokens(self,text):
		token_text = []
		doc = nlp(text.decode('utf-8'))
		for token in doc:
			token_text.append(token.text)
		return token_text

	def get_sentiment(self,text): #WIP
			sentiment_dict = json.loads(self.sentiment_nlp.annotate(text, properties=props))
			sentiment_tree = Tree.fromstring(sentiment_dict['sentences'][0]['sentimentTree'])


	def get_emphasis():
		#detect overuse of emojis
		pass

	def get_subjectivity(self,text):
		polartiy,subjectivity = TextBlob(text).sentiment
		return polarity, subjectivity

	def get_wh_words(self,text):
		wh_words = ["who","where","why","when","how","what","which","whose","whom","what kind","what time","how many","how much","how long","how often","how far","how old","how come"]
		for wh in wh_words:
			if wh in text.lower():
				return 1
		return 0


	"""
	context will always be history from previous two turns and query. It will be a list of three utterances
	"""

	def get_document_word_features(self,query,document,imdb_id): #query and document should be strings
		token_text, lemma, pos, ner, stop = self.get_token_lemma_pos_ner_stop(document,imdb_id)
		aspect_list, marker = self.check_aspect_word(token_text,ner) #incomplete
		#sentiment_list = get_sentiment_word_level()# incomplete
		sentiment_list = {}
		in_query_list = self.in_query(token_text,self.get_tokens(query))
		word_features = {}
		word_features = {'tokens':token_text, 'lemma':lemma, 'pos': pos, 'ner':ner, 'stop':stop,'aspects':aspect_list,'is_aspect':marker, 'sentiment':sentiment_list,'in_query':in_query_list} 
		return word_features

	def get_sentence_features(self,sentence): 
		sentence_features = {}
		polartiy,subjectivity = self.get_subjectivity(sentence)
		sentence_features = {'text':sentence, 'polartiy':polartiy, 'subjectivity': subjectivity}
		return sentence_features
		#return aspect, polartiy and sentence type

	def get_utterance_features(self,context,imdb_id): 
		query = context[-1]
		context = " ".join(i for i in context)
		token_text, lemma, pos, ner, stop = self.get_token_lemma_pos_ner_stop(context)
		aspect_list, marker = self.check_aspect_word(token_text,ner) #incomplete
		sentiment_list = self.get_sentiment_word_level()
		wh_presence = self.get_wh_words(query)
		utterances_features = {}
		utterances_features =  {'tokens':token_text, 'lemma':lemma, 'pos': pos, 'ner':ner, 'stop':stop,'aspects':aspect_list,'is_aspect':marker, 'sentiment':sentiment_list,'wh_presence':wh_presence} 
		return utterances_features


	def get_context_word_features(self,context,query,document,imdb_id): #context is a string
		token_text, lemma, pos, ner, stop = self.get_token_lemma_pos_ner_stop(context,imdb_id)
		aspect_list, marker = self.check_aspect_word(token_text,ner) #incomplete
		#sentiment_list = get_sentiment_word_level()# incomplete
		in_query_list = self.in_query(token_text,self.get_tokens(document))
		
		sentiment_list = {}
		word_features = {}
		word_features = {'tokens':token_text, 'lemma':lemma, 'pos': pos, 'ner':ner, 'stop':stop,'aspects':aspect_list,'is_aspect':marker, 'sentiment':sentiment_list,'in_query':in_query_list} 
		return word_features

	def get_context_features(self,context,context_aspect_list,gamma=0.8): #context should be a list here
		topic_tracker = np.zeros(len(aspect.keys()))
		polarity_1,_ = self.get_subjectivity(" ".join(i for i in context[::2]))
		if len(context) == 1:
			polarity_2 = 0.0
		else:
			polarity_2,_ = self.get_subjectivity(" ".join(i for i in context[1::2]))
		
		for ind_aspect_list in context_aspect_list:
			visited_aspects = list(set(ind_aspect_list))
			aspect_indices = [self.aspect_mapper[key] for key in visited_aspects]
			topic_tracker = gamma * topic_tracker
			for i in aspect_indices:
				topic_tracker[i] = 1 + topic_tracker[i]

		context_features = {'polartiy_1':polartiy_1,'polarity_2':polarity_2,'topic_tracker':topic_tracker}
			
		return context_features
	
