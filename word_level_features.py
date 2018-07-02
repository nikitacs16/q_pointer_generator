import json 
import numpy as np
import spacy
import string
from nltk.tree import Tree 
from stanfordcorenlp import StanfordCoreNLP
from textblob import TextBlob

sentiment_nlp = StanfordCoreNLP('/home/nikita/Downloads/stanford-corenlp-full-2018-02-27')
props = {'annotators':'sentiment','pipelineLanguage':'en','outputFormat':'json'}

nlp = spacy.load('en')
ner_aspect_mapper = {'ACTOR': 'acting' ,'DIRECTOR': 'direction', 'PRODUCER' : 'production', 'CINEMATOGRAPHER': 'cinematography' , 'CHARACTER': 'character'}
movie_kb = json.load(open('movie_kb.json'))
aspects = json.load(open('aspects.json'))
aspects_unigram = aspects[0]
aspects_bigram = aspects[1]
aspect_keys = aspects_unigram.keys()
aspect_mapper = {}
for pos,key in enumerate(aspect_keys):
	aspect_mapper[key] = pos
#pre_computed_tf_idf = #some file here

"""
Helper functions begin here
"""



def get_special_ner(word,curr_ner,imdb_id): #check if word is a special category word . Handcrafterd convert this to spacy code later
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

def get_unigram_aspect(word,word_ner):
		word = word.lower()
		for key in aspects_unigram.keys():
			if word in aspects_unigram[key]:
				return key, 1
		
		if word_ner in ner_aspect_mapper:
			return ner_aspect_mapper[word_ner], 1
		
		return "", 0

def get_bigram_aspect(bigram): 
	for key in aspects_bigram.keys():
		if bigram.lower() in aspects_bigram[key]:
			return key,key,1,1

	return "","",0,0		

def in_query(document_tokens,query_tokens):
	marker = np.zeros(len(document_tokens),dtype=int)
	query_tokens_l = [token.lower() for token in query_tokens]
	for j,token in enumerate(document_tokens):
		if token.lower() in query_tokens_l and token not in string.punctuation:
			marker[j] = 1
	return list(marker)

def check_aspect_word(tokens,tokens_ner): #binary bit
	marker = np.zeros(len(tokens),dtype=int)
	aspect_list = ["" for i in marker]
	for loc in range(len(tokens)-1):
		if marker[loc]==0:
			bigram_string = tokens[loc] + " " + tokens[loc+1]
			aspect_list[loc],aspect_list[loc+1],marker[loc],marker[loc+1] = get_bigram_aspect(bigram_string)
			
	for loc in range(len(tokens)):
		if marker[loc] == 0:
			aspect_list[loc], marker[loc] = get_unigram_aspect(tokens[loc], tokens_ner[loc]) 
				
	return aspect_list, list(marker)

def check_domain_word(tokens): #WIP in terms of data
	marker = np.zeros(len(tokens),dtype=int)
	for j,token in enumerate(tokens):
		if token.lower() in domain_words:
			marker[j] = 1
	return list(marker)

def check_genre_word(tokens,imdb_id): #WIP
	pass 

def get_token_lemma_pos_ner_stop(doc,imdb_id="",extended_ner=True):
	
	doc = nlp(doc)
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
			extended_ner.append(get_special_ner(word,word_ner,imdb_id))
		ner = extended_ner
	stop = list(np.array(stop) * 1)
	return token_text, lemma, pos, ner, stop

def get_tokens(text):
	token_text = []
	doc = nlp(text)
	for token in doc:
		token_text.append(token.text)
	return token_text

def get_sentiment(text): #WIP
		sentiment_dict = json.loads(sentiment_nlp.annotate(text, properties=props))
		sentiment_tree = Tree.fromstring(sentiment_dict['sentences'][0]['sentimentTree'])


def get_emphasis():
	#detect overuse of emojis
	pass

def get_subjectivity(text):
	polartiy,subjectivity = TextBlob(text).sentiment
	return polarity, subjectivity

def get_wh_words(text):
	wh_words = ["who","where","why","when","how","what","which","whose","whom","what kind","what time","how many","how much","how long","how often","how far","how old","how come"]
	for wh in wh_words:
		if wh in text.lower():
			return 1
	return 0


"""
context will always be history from previous two turns and query. It will be a list of three utterances
"""

def get_document_word_features(query,document,imdb_id): #query and document should be strings
	token_text, lemma, pos, ner, stop = get_token_lemma_pos_ner_stop(document,imdb_id)
	aspect_list, marker = check_aspect_word(token_text,ner) #incomplete
	#sentiment_list = get_sentiment_word_level()# incomplete
	sentiment_list = {}
	in_query_list = in_query(token_text,get_tokens(query))
	word_features = {}
	word_features = {'tokens':token_text, 'lemma':lemma, 'pos': pos, 'ner':ner, 'stop':stop,'aspects':aspect_list,'aspect_exists':marker, 'sentiment':sentiment_list,'in_query':in_query_list} 
	return word_features

def get_sentence_features(sentence): 
	sentence_features = {}
	polartiy,subjectivity = get_subjectivity(sentence)
	sentence_features = {'text':sentence, 'polartiy':polartiy, 'subjectivity': subjectivity}
	return sentence_features
	#return aspect, polartiy and sentence type

def get_utterance_features(context,imdb_id): 
	query = context[-1]
	context = " ".join(i for i in context)
	token_text, lemma, pos, ner, stop = get_token_lemma_pos_ner_stop(context)
	aspect_list, marker = check_aspect_word(token_text,ner) #incomplete
	sentiment_list = get_sentiment_word_level()
	wh_presence = get_wh_words(query)
	utterances_features = {}
	utterances_features =  {'tokens':token_text, 'lemma':lemma, 'pos': pos, 'ner':ner, 'stop':stop,'aspects':aspect_list,'aspect_exists':marker, 'sentiment':sentiment_list,'wh_presence':wh_presence} 
	return utterances_features


def get_context_word_features(context,query,document,imdb_id): #context is a string
	token_text, lemma, pos, ner, stop = get_token_lemma_pos_ner_stop(context,imdb_id)
	aspect_list, marker = check_aspect_word(token_text,ner) #incomplete
	#sentiment_list = get_sentiment_word_level()# incomplete
	in_query_list = in_query(get_tokens(query))
	sentiment_list = {}
	word_features = {}
	word_features = {'tokens':token_text, 'lemma':lemma, 'pos': pos, 'ner':ner, 'stop':stop,'aspects':aspect_list,'aspect_exists':marker, 'sentiment':sentiment_list} 
	return word_features

def get_context_features(context,context_aspect_list,gamma=0.8): #context should be a list here
	topic_tracker = np.zeros(len(aspect.keys()))
	polarity_1,_ = get_subjectivity(" ".join(i for i in context[::2]))
	if len(context) == 1:
		polarity_2 = 0.0
	else:
		polarity_2,_ = get_subjectivity(" ".join(i for i in context[1::2]))
	
	for ind_aspect_list in context_aspect_list:
		visited_aspects = list(set(ind_aspect_list))
		aspect_indices = [aspect_mapper[key] for key in visited_aspects]
		topic_tracker = gamma * topic_tracker
		for i in aspect_indices:
			topic_tracker[i] = 1 + topic_tracker[i]

	context_features = {'polartiy_1':polartiy_1,'polarity_2':polarity_2,'topic_tracker':topic_tracker}
		
	return context_features
	
