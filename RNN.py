import cProfile
from gensim.models import Word2Vec

cProfile.run("model = Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)")

cProfile.run("print model['computer']")
cProfile.run("print model['king']")
cProfile.run("")