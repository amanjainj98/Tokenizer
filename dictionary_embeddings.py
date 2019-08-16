import os
import json
import numpy as np
import string

def load_dictionary(dictionary_directory):
	data = dict()
	for filename in os.listdir(dictionary_directory):
		with open(os.path.join(dictionary_directory, filename)) as f:
			d = json.load(f)
			for key,value in d.items():
				key = key.lower()
				if value["MEANINGS"]:
					meanings = []
					for k,v in value["MEANINGS"].items():
						if v[2]:
							meanings.append(v[2])
						elif v[1]:
							meanings.append([v[1]])
					data[key] = meanings
					continue
				elif value["SYNONYMS"]:
					data[key] = [value["SYNONYMS"]]


	return data


def initialize_random(embedding_size):
	return np.random.rand(embedding_size)


def get_embedding_word(word,embeddings):
		word = word.lower()
		if word in embeddings:
			return np.mean(embeddings[word], axis=0)
		return None

def get_embedding_sentence(sentence,embeddings):
	translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 
	sentence = sentence.translate(translator)
	words = sentence.split()
	emb = []
	for word in words:
		e = get_embedding_word(word,embeddings)
		if e is not None:
			emb.append(e)

	if emb:
		emb = np.mean(np.array(emb),axis=0)
		return emb
	return None


def generate_dictionary_embeddings(dictionary_directory, embedding_size, num_iterations):
	data = load_dictionary(dictionary_directory)

	embeddings = dict()

	for key,value in data.items():
		embeddings[key] = np.array([initialize_random(embedding_size) for _ in range(len(value))])


	for _ in range(num_iterations):
		for key,value in data.items():
			for i in range(len(value)):
				all_se = []
				for v in value[i]:
					se =  get_embedding_sentence(v,embeddings)
					if se is not None:
						all_se.append(se)

				if all_se:
					embeddings[key][i] = np.mean(np.array(all_se),axis = 0)
	

	# for key, value in embeddings.items():
	# 	print(key,value)

	return embeddings





# embeddings = generate_dictionary_embeddings('data/',5,1)