# DocumentClustering

In this project I try to cluster documents by doing the following  

At present I use IMDB review dataset with to clusters for testing  

At first I create list of words (termins) from the data corpus 
Then I take each document, find every word from the termin list and create 
word embedding using [Bert](https://huggingface.co/docs/transformers/model_doc/bert) for every such word. After obtaining a list of embeddings I 
use [Hierarchical clustering algorithm](https://en.wikipedia.org/wiki/Hierarchical_clustering) in order to cluster 
the documents

At present there are available 2 jupyter notebooks  
- *select_words_tf_idf.ipynb*  
selects the termin
- *create_document_context_bert.ipynb*  
create document contexts and clusters  

Also there is alternative way to create termins:  
```>python tools/create_termins.py --num 50 --file ./termins.json```