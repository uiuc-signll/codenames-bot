from gensim.models import KeyedVectors

# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit= 200000)

def find_max(opposing, in_word):
    max = -1
    for opp_word in opposing:
        if model.similarity(opp_word, in_word) > max:
            max = model.similarity(opp_word, in_word)
    return max

def find_count(our_team, max_oppose, in_word):
    count = 0
    dict = {}
    for our_word in our_team:
        if model.similarity(our_word, in_word) > max_oppose:
            count += 1
            dict[our_word] = 1
    return count, dict

def find_clue(wordSet, our_team, opposing):
    curr_count = 0
    dict = {}
    out_word = ""
    for word in wordSet:
        opp_max = find_max(opposing, word)
        curr, d = find_count(our_team, opp_max, word)
        if curr >= curr_count:
            curr_count = curr
            out_word = word
            dict = d
    list = []
    for key in dict.keys():
        list.append(key)
    return out_word, list

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
clue, ret = find_clue(wordSet, redWords, blueWords)
print(clue)
print(ret)
