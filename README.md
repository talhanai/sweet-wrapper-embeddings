# sweet-wrapper-embeddings
This repo contains a wrapper script **GenerateEmbeddings.py** to train word or doc embeddings. To generate embeddings run:

```
text=mytextfile.txt
outputdir=mydir # directory to dump files to 
type=[par2vec|doc2vec] # type of embeddings to train
dim=100 # choose your value
nCpus=4 # number of CPUs
python GenerateEmbeddings.py $text $outputdir $type $dim $nCPUs
```
