from dictionary_embeddings import get_embedding_word, get_embedding_sentence, initialize_random
# from dictionary_embeddings import generate_dictionary_embeddings
import numpy as np

def generate_prefix_embeddings(prefix_file,embedding_size,embeddings):
	prefix_embeddings = dict()
	with open(prefix_file) as f:
		for line in f:
			if line[0] == '|' and line[1] == ' ':
				line = line[2:].split("||")
				p = line[0].strip()[:-1].lower()
				meanings = line[1].split(';')
				meanings = [m.strip()[1:-1].lower() for m in meanings]

				embs = [get_embedding_sentence(m,embeddings) for m in meanings]
				embs = [x for x in embs if x is not None]

				if embs:
					embs = np.array(embs)
					# emb = np.mean(embs,axis=0)
					prefix_embeddings[p] = embs

				else:
					prefix_embeddings[p] = np.expand_dims(initialize_random(embedding_size),axis=0)

	return prefix_embeddings


def generate_suffix_embeddings(suffix_file,embedding_size):
	suffix_embeddings = dict()
	with open(suffix_file) as f:
		for line in f:
			s = line.split()[0].strip()[1:]
			suffix_embeddings[s] = np.expand_dims(initialize_random(embedding_size),axis=0)

	return suffix_embeddings


def generate_stem_embeddings(stem_index,index_bases,embedding_size,embeddings):
	index_embeddings = dict()
	for key,value in index_bases.items():
		embs = [embeddings[v] for v in value if v in embeddings]
		if embs:
			embs = np.array(embs)
			index_embeddings[key] = embs

		else:
			index_embeddings[key] = np.expand_dims(initialize_random(embedding_size),axis=0)

	return index_embeddings

# embeddings = generate_dictionary_embeddings('data/',5,1)
# prefix_embeddings = generate_prefix_embeddings('prefixes.txt',5,embeddings)
# suffix_embeddings = generate_suffix_embeddings('suffixes.txt',5)

# for k,v in suffix_embeddings.items():
# 	print(k,v)