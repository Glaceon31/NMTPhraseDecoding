import sys

src_one = open(sys.argv[1], 'r').read().split('\n')[0]
trg = open(sys.argv[2], 'r').read().split('\n')
if trg[-1] == '':
    del trg[-1]

src_multi = open(sys.argv[3], 'w')
src_multi.write('\n'.join([src_one]*len(trg)))
src_multi.close()
