from generate_stems import generate_stems
from dictionary_embeddings import generate_dictionary_embeddings
from generate_prefix_stem_suffix_embeddings import generate_prefix_embeddings, generate_suffix_embeddings, generate_stem_embeddings
import numpy as np
import pickle


lemma_file = 'lemma.en.txt'
dictionary_data = 'data/'
prefix_file = 'prefixes.txt'
suffix_file = 'suffixes.txt'
vocab_file = 'vocab.txt'
embeddings_file = 'embeddings'

embedding_size = 5
num_iterations = 10

stem_index,index_bases = generate_stems(lemma_file)
dictionary_embeddings = generate_dictionary_embeddings('data/',embedding_size,num_iterations)
prefix_embeddings = generate_prefix_embeddings(prefix_file,embedding_size,dictionary_embeddings)
suffix_embeddings = generate_suffix_embeddings(suffix_file,embedding_size)
index_embeddings = generate_stem_embeddings(stem_index,index_bases,embedding_size,dictionary_embeddings)

embeddings = []
index = 0
for k in sorted(index_embeddings):
	embeddings.append(index_embeddings[k])

with open(vocab_file,'w') as f:

	for k,v in stem_index.items():
		f.write(k+"		,	"+str(v)+"\n")		
		index+=1


	for k,v in prefix_embeddings.items():
		f.write("**"+k+"		,	"+str(index)+"\n")
		embeddings.append(v)
		index+=1

	for k,v in suffix_embeddings.items():
		f.write("##"+k+"		,	"+str(index)+"\n")
		embeddings.append(v)
		index+=1

embeddings = np.array(embeddings)
pickle.dump(embeddings, open(embeddings_file, "wb"), protocol=2)