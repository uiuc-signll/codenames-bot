from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)

words = [

'kangaroo', 'mouse', 'cell', 'limousine', 'doctor',
'helicopter', 'moon', 'plate', 'bell', 'iron',
'roulette', 'day', 'fish', 'teacher', 'brush',
'dress', 'light', 'triangle', 'Europe', 'mug',
'hospital', 'olive', 'horseshoe', 'boom', 'force',
'Germany', 'relativity', 'energy', 'nuclear', 'physics', 'Nobel', 'Berlin',
]


#from gensim.models import Word2Vec
# model = Word2Vec.load(path/to/your/model)

#print(model.similarity('France', 'Spain'))

while True:
    clue = input('What was the clue? ')
    cosines = []
    for word in words:
        cosines.append((model.similarity(clue, word), word))

    cosines.sort()
    cosines.reverse()

    for i in cosines[:]:
        print(i)
    print()

#for i in result:
#    print(i)
