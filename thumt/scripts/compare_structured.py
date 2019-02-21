from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import numpy 
import time
import math
import json

from calculate_oracle import bleu

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--hypos", type=str, required=True, nargs="+")
    parser.add_argument("--names", type=str, required=True, nargs="+")
    parser.add_argument("--refs", type=str, required=True, nargs="+")
    parser.add_argument("--senid", type=int, default=-1)
    parser.add_argument("--hidden", action="store_true")
    return parser.parse_args()

def rbpe(inp):
    return inp.replace('@@ ', '')

def getlines(inp):
    result = inp.split('\n')
    if result[-1] == '':
        del result[-1]
    return result

def splitline(line):
    result = line.split()
    pos = 0
    while pos < len(result):
        if result[pos][0] == '<' and result[pos][-1] != '>':
            if pos+1 < len(result):
                result[pos] += ''+result[pos+1]
                del result[pos+1]
            else:
                print('wrong!! ', result[pos])
                exit()
        else:
            pos += 1
    return result


if __name__ == "__main__":
    args = parseargs()

    src = open(args.src, 'r').read() 
    hypos = [open(hypo, 'r').read() for hypo in args.hypos]
    refs = [open(ref, 'r').read() for ref in args.refs]

    if args.senid == -1:
        lsrc = getlines(src)
        lhypos = [getlines(h) for h in hypos]
        lrefs = [getlines(r) for r in refs]
    else:
        lsrc = [getlines(src)[args.senid]]
        lhypo = [getlines(hypo)[0]]
        if args.compare:
            lcompare = [getlines(compare)[args.senid]]
        #loracle = [getlines(oracle)[args.senid]]
        lbase = [getlines(baseline)[args.senid]]
        lrefs = [[getlines(r)[args.senid]] for r in refs]

    print(len(lsrc))

    worse = 0
    better = 0
    for i in range(len(lsrc)):
        reftmp = [' '.join(splitline(t[i])) for t in lrefs]
        hypotmp = [' '.join(splitline(t[i])) for t in lhypos] 
        bleus = [0]*len(lhypos)
        bleus_verbose = ['']*len(lhypos)
        for j in range(len(lhypos)):
            bleus[j] = bleu(rbpe(hypotmp[j]), reftmp, 4, verbose=False)
            bleus_verbose[j] = bleu(rbpe(hypotmp[j]), reftmp, 4, verbose=True)
        if args.hidden:
            equal = True
            b = bleus[0]
            if j in range(1, len(lhypos)):
                if b != bleus[j]:
                    equal = False
            if equal:
                continue
        print('=== sentence #%d ===' %i)
        print('src:', lsrc[i])
        for r in range(len(reftmp)):
            print('ref'+str(r)+':', reftmp[r])
        results = sorted(zip(args.names, hypotmp, bleus, bleus_verbose), key=lambda x:x[2])
        for j in range(len(lhypos)):
            result = results[j]
            print('(#'+str(len(lhypos)-j)+')', result[0]+':', result[1], '('+result[3]+')')
        #print('ours:', lhypo[i], '('+str(bleu(rbpe(lhypo[i]), reftmp, 4, verbose=True))+')')
        print('\n')
        #if i == 1:
        #    break

