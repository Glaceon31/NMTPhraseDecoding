import sys

pt = open(sys.argv[1], 'r')
inp = open(sys.argv[2], 'r').read()
ngrams = 10

phrases = {}
lines_inp = inp.split('\n')
for line in lines_inp:
    words = line.split(' ')
    lens = len(words)
    for start in range(lens):
        for end in range(start+1, min(start+ngrams, lens)):
            phrase = ' '.join(words[start:end])
            if phrases.has_key(phrase):
                phrases[phrase] += 1
            else:
                phrases[phrase] = 1

print('start pt')
result = []
line = pt.readline()
count = 0
num = 0
while line:
    if (count+1) % 1000000 == 0:
        print(num, '/', count+1)
    src, trg, probs, align, counts, _ = line.split(' ||| ')
    if phrases.has_key(src):
        result.append(line)
        num += 1
    count += 1
    line = pt.readline()
output = open(sys.argv[3], 'w')
output.write(' '.join(result))
output.close()

