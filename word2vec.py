from gensim.models import word2vec #word2vec is a package we can use
import logging

PATH2SOGOUT = '' #file path
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence(PATH2SOGOUT) #one line is one sentence
model = word2vec.Word2Vec(sentences, workers=20, sg=1, size=300, min_count=50)  #Train, use and evaluate neural networks 
model.wv.save_word2vec_format('300Tvectors.txt', binary=False) # Store the model with txt format(?)
#The trained word vectors can also be stored/loaded from a format compatible with the original word2vec implementation
