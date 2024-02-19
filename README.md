# DocumentClustering

In this project I try to cluster documents by doing the following  

At present I use IMDB review dataset with to clusters for testing  

At first I create list of words (termins) from the data corpus 
Then I take each document, find every word from the termin list and create 
word embedding using [Bert](https://huggingface.co/docs/transformers/model_doc/bert) for every such word. After obtaining a list of embeddings I 
use [Hierarchical clustering algorithm](https://en.wikipedia.org/wiki/Hierarchical_clustering) in order to cluster 
the documents

There are two availabele tools:
- **tools/create_termins.py**  
usage example:  
```>python tools/create_termins.py --num 50 --file termins.json --lang en --stopwords ./stopwords/en.json --dataset ./doc.parquet```  
- **tools/make_clusters.py**  
usage example:  
```>python tools/make_clusters.py --dataset dataset_ua.parquet --termins ./termins_ua.json --lang uk --max_samples 100 --seed 0 --log_res ./log_res.txt --log_level 2 --clusters 7 --cuda 0```

