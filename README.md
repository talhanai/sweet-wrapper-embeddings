# sweet-wrapper-embeddings
This repo contains a wrapper script **GenerateEmbeddings.py** to train word or doc embeddings.

The script expects the following arguments
- **text** file.
- **output directory** to dump the results as **_embeddigns.npz*.
- **type** of embeddings to train ('word2vec' or 'doc2vec').
- **dimension** size of embeddings
- **Number of CPUS** to parallelize the process. *Note that if you use more than a single CPU for training then the random initializations may not be the same if you ran the script twice, therefore your resulting embeddings may not be the same. This can be an issue if you care to reproduce results.*

### Word Embeddings
To generate word embeddings run:

```
python GenerateEmbeddings.py $text $outputdir word2vec $dim $nCPUs
```

### Doc Embeddings
To generate embeddings over a sequence of words (from the beginning '^' to the end of a line '$') run:

```
python GenerateEmbeddings.py $text $outputdir doc2vec $dim $nCPUs
```

### References
This script was put together based on [this for word embeddings](https://gist.github.com/codekansas/15b3c2a2e9bc7a3c345138a32e029969) and [this for doc embeddings](http://linanqiu.github.io/2015/10/07/word2vec-sentiment/).
