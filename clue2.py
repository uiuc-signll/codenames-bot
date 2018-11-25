from gensim.models import KeyedVectors

# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(filename, binary = True, limit = 200000)
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
max = 0
output = ""
out_list = []
for i in range(2<<len(blueWords)-1, -1, -1):
    temp = []
    for j in range(0, len(blueWords)):
        if i&(1<<j) > 0:
            temp.append(blueWords[j])
    if len(temp) == 0 or len(temp) == 1:
        continue
    x, y = model.most_similar_cosmul(positive=temp)[0]
    flag = False
    for word in temp:
        if word in x:
            flag = True
    if flag:
        continue
    if y >= max:
        max = y
        output = x
        out_list = temp
print(out_list)
print(output)
print(max)
