from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=18000)

wordSet = model.wv.vocab.keys()

wordVectorDict = {}


for word in wordSet:
    wordVectorDict[word] = [float(i) for i in model[word]]

import json
# print(json.dumps(wordVectorDict, indent=2))

with open('wordvectors.json', 'w') as file:
    json.dump(wordVectorDict, file, indent=2)

print(model.similarity('for', 'that'))
import numpy as np
from scipy import spatial
print(np.dot(model['for'], model['that'])/(np.linalg.norm(model['for']) * np.linalg.norm(model['that'])))
print(1 - spatial.distance.cosine(model['for'], model['that']))
print()
print(model.similarity('amazing', 'great'))

