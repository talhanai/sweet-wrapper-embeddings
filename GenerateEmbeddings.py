#!/usr/bin/python

# CREATING EMBEDDINGS OF TEXT
#
# To Run: 
# python GenerateEmbeddings.py transcripts/transcript.all.txt transcripts/exp/ word2vec 50 5
# python GenerateEmbeddings.py textfile outputdir [word2vec|par2vec] embeddings_size nCPUs_to_parallel


import json
import os
import numpy as np
import sys

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import pylab as Plot
import scipy.io as sio
import time


# UNPLUG BUFFER TO PRINT TO STDOUT
# this is so that we can print in slurm mode
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

# update stdout style
sys.stdout = Unbuffered(sys.stdout)

# tokenizer: can change this as needed
tokenize = lambda x: simple_preprocess(x)


#########################################################################################################
# CREATES WORD EMBEDDINGS
# CODE FROM: https://gist.github.com/codekansas/15b3c2a2e9bc7a3c345138a32e029969
#########################################################################################################
def create_embeddings(data_file, embeddings_path='word_embeddings.npz', vocab_path='map.json', **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """

    # parses sentences
    class SentenceGenerator(object):
        def __init__(self, filename):
            self.filename = filename

        def __iter__(self):
            # for fname in os.listdir(self.dirname):
            for line in open(self.filename):
            	# filtered_line = [word for word in line.split(' ') if word not in stopwords.words('english')]
            	# print line
                yield tokenize(line)

    print("==== Processing Sentences ...")
    sentences = SentenceGenerator(data_file)

    print("==== Modeling Embeddings ...")
    model 	= Word2Vec(sentences, **params)

    print("==== Saving File ...")
    model.save(embeddings_path)


#########################################################################################################
# CREATE DOC EMBEDDINGS
# FROM: http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
#########################################################################################################
def create_embeddings_doc(data_file, embeddings_path='doc_embeddings.npz', vocab_path='map.json', **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """

    # parses sentences
    class LabeledLineSentence(object):
        def __init__(self, filename):
            self.filename = filename
        def __iter__(self):
            for uid, line in enumerate(open(self.filename)):

                # what is this tagging doing? can't remember
                yield TaggedDocument(words=line.replace('(','').replace(')','').split(), tags=['SENT_%s' % uid])

    print("==== Processing Sentences ...")
    sentences = LabeledLineSentence(data_file)

    print("==== Modeling Embeddings ...")
    model 	= Doc2Vec(sentences, **params)

    print("==== Saving File ...", embeddings_path)
    model.save(embeddings_path)


#########################################################################################################
# MAIN
#########################################################################################################
if __name__ == "__main__":


	# tracking time it takes to do processes
	sTime = time.time()

	# INPUT VARIABLES 
	# --------------------------------------------
	# source text file with complete transcript
	# data_file = 'transcripts/transcript.all.txt'
	data_file = sys.argv[1]
	print("==== Data File                 : " + data_file)
    
	outdir    = sys.argv[2]
	print("==== Files will be saved to    : " + outdir)

	# define here what kind of embeddings you want to generate
	# embeddingsType = 'word2vec'
	# embeddingsType = 'par2vec'
	embeddingsType = sys.argv[3]

	# size of embeddings we want to train
	# embeddings_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

	embeddings_sizes = int(sys.argv[4])
	i = embeddings_sizes
	print("==== Processing Embeddings Size: ", str(i) )


	# number of cpus to use
	nCPUs = int(sys.argv[5])
	try:
		min_count = int(sys.argv[6])
		window = int(sys.argv[7])
		epochs = int(sys.argv[8])
	except:
		print('  EXCEPT: missing some arguments, min_count, window, epochs')
		min_count = 1
		window = 5
		epochs = 25


	# if we want to generate PARAGRAPH embeddings
	# -------------------------------------------
	if embeddingsType == 'par2vec':

		# define the files to read and write to
		embeddings_path = outdir + 'doc_embeddings.' + str(i) + '.npz'
		vocab_path 		= outdir + 'map.json'

		print('==== Creating Embeddings         ' + 'Par2Vec ...')
		# model the doc embeddings
		create_embeddings_doc(data_file=data_file, 
	 						embeddings_path=embeddings_path,
	 						vocab_path=vocab_path,
	 						vector_size=i, min_count=min_count, window=window, workers=nCPUs, seed=1, sample=1e-4, epochs=epochs)

		eTime = time.time()
		print("toc: ", str(eTime - sTime))

		# load the doc embeddings (optional)
		# model = Doc2Vec.load(embeddings_path)


	# if we want to generate WORD embeddings
	# --------------------------------------
	elif embeddingsType == 'word2vec':

		# define the files to read and write to
		embeddings_path = outdir + 'word_embeddings.' + str(i) + '.npz'
		vocab_path 		= outdir + 'map.json'

		print('==== Creating Embeddings         ' + 'Word2Vec ...')
		# create word embeddings
		create_embeddings(data_file=data_file, 
						embeddings_path=embeddings_path,
						vocab_path=vocab_path,
						size=i, min_count=5, window=5, workers=nCPUs, seed=1, sg=1, iter=25)

		eTime = time.time()
		print("toc: ", str(eTime - sTime))

	eTime = time.time()
	print("toc: ", str(eTime - sTime))


# EOF
