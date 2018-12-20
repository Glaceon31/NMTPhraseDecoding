import sys

pt = open(sys.argv[1], 'r').read()

def getinformation(a):
    src, trg, probs, align, counts, _ = a
    freq = int(counts.split(' ')[-1])
    prob = sum([float(i) for i in probs.split(' ')])
    length = len(src.split(' '))
    return [freq, prob, length]

def compare(a, b):
    if a[-3] > b[-3]:
        return 1
    elif a[-3] < b[-3]:
        return -1
    if a[-2] > b[-2]:
        return 1
    elif a[-2] < b[-2]:
        return -1
    if a[-1] > b[-1]:
        return 1
    elif a[-1] < b[-1]:
        return -1
    return 0
    

items = pt.split('\n')
if items[-1] == '':
    del items[-1]
items = [i.split(' ||| ') for i in items]
result = []
for i in items:
    if len(i[0].split(' ')) > 2:
        result.append(i)
items = result
items = [i+getinformation(i) for i in items]
print(len(items))

print('start sorting')
sorted_items = sorted(items, reverse=True, cmp=compare)
sorted_items = [' ||| '.join(i[:-3]) for i in sorted_items]
output = open(sys.argv[2], 'w')
output.write('\n'.join(sorted_items))
output.close()
