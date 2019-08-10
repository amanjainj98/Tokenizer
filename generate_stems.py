from nltk import PorterStemmer

def generate_all_derivations(lemma_file):
	base_dervs = dict()

	with open(lemma_file,'r') as f:
		for line in f:
			line = line.split()
			base = line[0].split('/')
			base = base[0]
			dervs = line[2].split(',')

			base = base.lower()
			if not base.isalpha():
				continue

			if len(base) < 2:
				continue

			base_dervs[base] = [derv for derv in dervs if derv.isalpha() and derv != base]


	# for key, value in base_dervs.items():
	# 	print(key,value)

	visited = set()
	
	def find_all_dervs(root):

		if root not in base_dervs:
			return []

		ans = base_dervs[root]

		for derv in base_dervs[root]:
			if not derv in visited:
				visited.add(derv)
				ans.extend(find_all_dervs(derv))


		return list(set(ans))


	base_all_dervs = dict()

	for key, value in base_dervs.items():
		if key not in visited:
			visited.add(key)
			all_dervs = find_all_dervs(key)
			if all_dervs:
				base_all_dervs[key] = all_dervs

	return base_all_dervs
	


def generate_stems(lemma_file):
	base_all_dervs = generate_all_derivations(lemma_file)
	stemmer = PorterStemmer()
	
	stem_index = dict()
	index_bases = dict()
	index = 0

	for key, value in base_all_dervs.items():
		v = value
		v.append(key)
		stems = list(set([stemmer.stem(derv) for derv in v]))
		flag = False
		for stem in stems:
			if stem not in stem_index:
				flag = True
				stem_index[stem] = index
			else:
				i = stem_index[stem]
				if key not in index_bases[i]:
					index_bases[i].append(key)

		if flag:
			index_bases[index] = [key]
			index+=1


	return stem_index,index_bases


stem_index,index_bases = generate_stems('lemma.en.txt')
for key, value in index_bases.items():
		print(key,value)
