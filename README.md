# Tokenizer

### How to run

- To create vocabulary and embeddings run
```shell
$ python3 create_vocab.py <embedding_size> <num_iterations>
```
Where 
embedding_size is the size (dimensions) of embeddings to be generated (eg - 128),
num_iterations is the number of iterations to be run on dictionary to generate embeddings (eg - 10)

This will create two files - vocab.txt and embeddings (pickle dump)


- To test the tokenizer run
```shell
$ python3 run_tokenizer.py <path/to/file/to/tokenize> 
```
This will print each line and its tokenization on screen, redirect it to a file for better readability

---

### Refrences
- Dictionary - `https://github.com/tusharlock10/Dictionary/blob/master/data.7z`
- Lemma list - `https://github.com/skywind3000/lemma.en`
- Suffixes - `https://en.wiktionary.org/wiki/Appendix:English_suffixes`
- Prefix - `https://en.wikipedia.org/wiki/Prefix`
