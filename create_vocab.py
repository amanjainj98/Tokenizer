from generate_stems import generate_stems
from dictionary_embeddings import generate_dictionary_embeddings, initialize_random
from generate_prefix_stem_suffix_embeddings import generate_prefix_embeddings, generate_suffix_embeddings, generate_stem_embeddings
import numpy as np
import pickle

import argparse
import sys

lemma_file = 'lemma.en.txt'
dictionary_data = 'data/'
prefix_file = 'prefixes.txt'
suffix_file = 'suffixes.txt'
vocab_file = 'vocab.txt'
embeddings_file = 'embeddings'


try:
	parser = argparse.ArgumentParser()
	parser.add_argument("embedding_size", help="Size (dimensions) of embeddings to be generated (eg - 128)",type=int)
	parser.add_argument("num_iterations", help="Number of iterations to be run on dictionary to generate embeddings (eg - 10)",type=int)
	args = parser.parse_args()
	print("Embedding size - " + str(sys.argv[1]) +"\nNumber of itearions - " + str(sys.argv[2]) + "\n")
except:
	e = sys.exc_info()[0]
	print(e)
	exit()


embedding_size = args.embedding_size
num_iterations = args.num_iterations

stem_index,index_bases = generate_stems(lemma_file)
dictionary_embeddings = generate_dictionary_embeddings('data/',embedding_size,num_iterations)
prefix_embeddings = generate_prefix_embeddings(prefix_file,embedding_size,dictionary_embeddings)
suffix_embeddings = generate_suffix_embeddings(suffix_file,embedding_size)
index_embeddings = generate_stem_embeddings(stem_index,index_bases,embedding_size,dictionary_embeddings)

embeddings = []
index = len(index_embeddings)
for k in sorted(index_embeddings):
	embeddings.append(index_embeddings[k])

special_tokens = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','[NUM]']
punctuations = ['!','\"','#','$','&','\'','(',')',',','-','.','/',':',';','?','@','[','\\',']','_','`']
mathematical_symbols = ['%','*','+','<','=','>','^']
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now"]
with open(vocab_file,'w') as f:

	for k,v in stem_index.items():
		f.write(k+"		,	"+str(v)+"\n")	


	for s in stop_words:
		if s not in stem_index:
			f.write(s+"		,	"+str(index)+"\n")
			embeddings.append([initialize_random(embedding_size)])
			index+=1	


	for k,v in prefix_embeddings.items():
		f.write("**"+k+"		,	"+str(index)+"\n")
		embeddings.append(v)
		index+=1

	for k,v in suffix_embeddings.items():
		f.write("##"+k+"		,	"+str(index)+"\n")
		embeddings.append(v)
		index+=1

	for s in special_tokens:
		f.write(s+"		,	"+str(index)+"\n")
		embeddings.append([initialize_random(embedding_size)])
		index+=1

	for s in punctuations:
		f.write(s+"		,	"+str(index)+"\n")
	embeddings.append([initialize_random(embedding_size)])
	index+=1

	for s in mathematical_symbols:
		f.write(s+"		,	"+str(index)+"\n")
	embeddings.append([initialize_random(embedding_size)])
	index+=1



embeddings = np.array(embeddings)
pickle.dump(embeddings, open(embeddings_file, "wb"), protocol=2)


print("Vocab size - " + str(index) + "\n")