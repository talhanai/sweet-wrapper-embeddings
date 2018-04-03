# sweet-wrapper-embeddings
This repo contains a wrapper script **GenerateEmbeddings.py** to train word or doc embeddings. 

To generate word embeddings run:

```
python GenerateEmbeddings.py $text $outputdir word2vec $dim $nCPUs
```

To generate embeddings over a sequence of words (from the beginning '^' to the end of a line '$', i.e. a sentence, paragraph, document) run:

```
python GenerateEmbeddings.py $text $outputdir doc2vec $dim $nCPUs
```
