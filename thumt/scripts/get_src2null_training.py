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
    msg = "get training pair for src2null"
    usage = "get_src2null_training.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--alignments", type=str, nargs="+", required=True,
                        help="alignment files")
    parser.add_argument("--output", type=str, help="output path")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    source = open(args.source, 'r')
    output = open(args.output, 'w')
    alignments = []
    for i in range(len(args.alignments)):
        alignments.append(open(args.alignments[i], 'r'))

    aligns = []
    results = []
    count = 0
    src = source.readline()
    for i in range(len(args.alignments)):
        aligns.append(alignments[i].readline()) 
    while src:
        count += 1
        if count % 10000 == 0:
            print(count)
        length = len(src.split(' '))
        notempty_count = [0] * length
        for i in range(len(args.alignments)):
            tmp_not_empty = [0] * length
            tmpaligns = aligns[i].split(' ')
            for a in tmpaligns:
                if len(a.split('-')) != 2:
                    continue
                srcpos, trgpos = a.split('-') 
                tmp_not_empty[int(srcpos)] = 1
            notempty_count = [notempty_count[i]+tmp_not_empty[i] for i in range(length)]
        notempty_count = [notempty_count[i]*1.0/5.0 for i in range(length)]
        output.write(' '.join([str(notempty_count[i]) for i in range(length)])+'\n')

        src = source.readline()
        aligns = []
        for i in range(len(args.alignments)):
            aligns.append(alignments[i].readline()) 

