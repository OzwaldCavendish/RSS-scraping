"""

Martin Wood - 14/07/2019

For convenience; wraps up a lot of the paraphenalia involved in 
getting word/sentence embeddings using various pre-trained models.

"""

import torch
import pickle
import nltk

import numpy as np

from gensim.models import Doc2Vec
from nltk.corpus import stopwords

# Local import, Requires local copy of InferSent model code with base
# word2vec models and such copied to correct locations
from InferSent.models import InferSent


class InferSentModel():
	"""
	Encapsulates the entire setup process and default configuration for
	loading a pre-trained InferSent document embeddings model and
	calculating the embeddings for a given corpus.
	"""
	
	# For InferSent sentence level encoder
	params_model = {'bsize': 64,
					'word_emb_dim': 300,
					'enc_lstm_dim': 2048,
					'pool_type': 'max',
					'dpout_model': 0.0}
	
	def __init__(self,
				 sentences,
				 labels,
				 MODEL_PATH = './InferSent/encoder/infersent2.pkl',
				 W2V_PATH = './InferSent/dataset/fastText/crawl-300d-2M.vec',
				 params_model = {}):
		
		self.MODEL_PATH = MODEL_PATH
		self.W2V_PATH = W2V_PATH
		
		for key in params_model.keys():
			self.params_model[key] = params_model[key]
		
		# Save the sentence labels
		self.labels = labels
		
		# Configure the actual model
		self.model = InferSent(self.params_model)
		self.model.load_state_dict(torch.load(self.MODEL_PATH))
		self.model.set_w2v_path(self.W2V_PATH)
		self.model.build_vocab(sentences, tokenize=True)
		
		self.core_embeddings = self.model.encode(sentences, tokenize=True)
	
	
	def get_embeddings(self, labels=True):
		"""
		Convenience function for getting the embeddings as an array or
		with the labels.
		"""
		if labels:
			return {x[0]:x[1] for x in zip(self.labels, self.core_embeddings)}
		else:
			return self.core_embeddings
		
		
	def get_more_embeddings(self, new_sentences, new_labels=None, labels=True):
		"""
		Get embeddings for sentences not in the original set. These are
		not stored in the object, merely returned.
		"""
		if labels:
			return {x[0]:x[1] for x in zip(new_labels, self.model.encode(new_sentences, tokenize=True))}
		else:
			return self.model.encode(new_sentences, tokenize=True)


class GloveWordModel():
	"""
	Encapsulates load and setup process for GloVE word embedding model
	with summing of vectors over text.
	"""
	
	def __init__(self,
				 sentences,
				 labels,
				 MODEL_PATH = "./models/glove.6B.100d.txt"):
		
		# Load the word-vector lookup table
		self.word_embeddings = {}
		with open(MODEL_PATH, encoding="utf-8") as f:
			for line in f:
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				self.word_embeddings[word] = coefs
		
		self.labels = labels
		nltk.download('stopwords')
		
		self.core_embeddings = self.get_summed_word_vectors(sentences)
	
	def clean_sentence(self, sentence, remove_stopwords=True):
		""" Utility, clean brutally. """
		sentence = sentence.replace('[^a-zA-Z]', ' ').lower()
			
		if remove_stopwords:
			sentence = " ".join([word for word in sentence.split() if word not in stopwords.words('english')])
		
		return sentence
		
	def get_summed_word_vectors(self, sentences):
		""" Creates summed word vectors for each sentence. """
		embeddings = []
		
		for s in sentences:
			if len(s) != 0:
				cleaned = self.clean_sentence(s)
				v = sum([self.word_embeddings.get(w, np.zeros((100,))) for w in cleaned.split()]) / ( len(cleaned.split()) + 0.001 )
			
			else:
				v = np.zeros((100, 0))
			embeddings.append(v)
		
		return np.asarray(embeddings)
			
	def get_embeddings(self, labels=True):
		"""
		Convenience function for getting the embeddings as an array or
		with the labels.
		"""
		if labels:
			return {x[0]:x[1] for x in zip(self.labels, self.core_embeddings)}
		else:
			return self.core_embeddings
	
	def get_more_embeddings(self, new_sentences, new_labels=None, labels=True):
		"""
		Get embeddings for sentences not in the original set. These are
		not stored in the object, merely returned.
		"""
		if labels:
			return {x[0]:x[1] for x in zip(new_labels, self.get_summed_word_vectors(new_sentences))}
		else:
			return self.model.encode(new_sentences, tokenize=True)
