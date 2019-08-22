import tokenization
import argparse
import sys

try:
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file", help="Path of file to tokenize",type=str)
	args = parser.parse_args()
except:
	e = sys.exc_info()[0]
	print(e)
	exit()

input_file = args.input_file
all_tokens = []

tokenizer = tokenization.FullTokenizer(
      vocab_file='vocab.txt', do_lower_case=True)

with open(input_file, "r") as reader:
  while True:
    line = tokenization.convert_to_unicode(reader.readline())
    if not line:
      break
    line = line.strip()

    tokens = tokenizer.tokenize(line)
    if tokens:
      all_tokens.append(tokens)
      print (line,tokens)