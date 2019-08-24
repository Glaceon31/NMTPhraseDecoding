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
    parser.add_argument("--hypo", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--compare", type=str)
    parser.add_argument("--hidden", action="store_true")
    #parser.add_argument("--oracle", type=str, required=True)
    parser.add_argument("--refs", type=str, required=True, nargs="+")
    parser.add_argument("--senid", type=int, default=-1)
    return parser.parse_args()

def rbpe(inp):
    return inp.replace('@@ ', '')

def getlines(inp):
    result = inp.split('\n')
    if result[-1] == '':
        del result[-1]
    return result

if __name__ == "__main__":
    args = parseargs()

    src = open(args.src, 'r').read() 
    hypo = open(args.hypo, 'r').read() 
    if args.compare:
        compare = open(args.compare, 'r').read()
    #oracle = open(args.oracle, 'r').read() 
    baseline = open(args.baseline, 'r').read()
    refs = [open(ref, 'r').read() for ref in args.refs]

    if args.senid == -1:
        lsrc = getlines(src)
        lhypo = getlines(hypo)
        #loracle = getlines(oracle)
        if args.compare:
            lcompare = getlines(compare)
        lbase = getlines(baseline)
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
        reftmp = [t[i] for t in lrefs]
        if args.hidden and args.compare and lhypo[i] == lcompare[i]:
            continue
        print('=== %d ===' %i)
        print('src:', lsrc[i])
        print('ours:    ', lhypo[i], '('+str(bleu(rbpe(lhypo[i]), reftmp, 4, verbose=True))+')')
        print('baseline:', lbase[i], '('+str(bleu(rbpe(lbase[i]), reftmp, 4, verbose=True))+')')
        if args.compare:
            print('compare: ', lcompare[i], '('+str(bleu(rbpe(lcompare[i]), reftmp, 4, verbose=True))+')')
        #print('oracle:', loracle[i], '('+str(bleu(rbpe(loracle[i]), reftmp, 4, verbose=True))+')')
        if bleu(rbpe(lhypo[i]), reftmp, 4, verbose=True) > bleu(rbpe(lcompare[i]), reftmp, 4, verbose=True):
            better += 1
        elif bleu(rbpe(lhypo[i]), reftmp, 4, verbose=True) < bleu(rbpe(lcompare[i]), reftmp, 4, verbose=True):
            worse += 1
        for r in range(len(reftmp)):
            print('ref'+str(r)+':', reftmp[r])
        print('\n')
        #if i == 100:
        #    break
    print('better:', better)
    print('worse:', worse)

