#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os
import json


def parseargs():
    msg = "Generate phrase table"
    usage = "phrase_table.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--input", type=str, required=True,
                        help="origin phrase-table.gz from Moses")
    parser.add_argument("--mincount", type=int, default=5)
    parser.add_argument("--minprob", type=float, default=0.01)
    parser.add_argument("--output", type=str, help="output path")
    
    return parser.parse_args()

def valid(probs, counts, args):
    p = probs.split(' ')
    c = counts.split(' ')
    p = [float(i) for i in p]
    c = [int(float(i)) for i in c]
    if c[0] < args.mincount or c[1] < args.mincount or c[2] < args.mincount:
        return False
    if p[2] < args.minprob:
        return False
    return True

if __name__ == "__main__":
    args = parseargs()
    content = open(args.input, 'r').read()
    lines = content.split('\n')
    if lines[-1].strip() == '':
        del lines[-1]
    result = {}
    count = 0
    ok = 0
    for line in lines:
        count += 1
        src, trg, probs, align, counts, _ = line.split(' ||| ')
        if valid(probs, counts, args):
            ok += 1
            if result.has_key(src):
                result[src].append(trg)
            else:
                result[src] = [trg]
        if count % 100000 == 0:
            print(ok, '/', count)
    json.dump(result, open(args.output, 'w'))


