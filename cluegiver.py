from gensim.models import KeyedVectors
# load the google word2vec model
filename = '../GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)

words = [
'kangaroo', 'mouse', 'cell', 'limousine', 'doctor',
'helicopter', 'moon', 'plate', 'bell', 'iron',
'roulette', 'day', 'fish', 'teacher', 'brush',
'dress', 'light', 'triangle', 'Europe', 'mug',
'hospital', 'olive', 'horseshoe', 'boom', 'force',
]

labels = "RRRRRRRRRBBBBBBBBNNNNNNNA"


redWords = [words[i] for i in range(25) if labels[i] == 'R']
blueWords = [words[i] for i in range(25) if labels[i] == 'B']


wordSet = model.wv.vocab.keys()