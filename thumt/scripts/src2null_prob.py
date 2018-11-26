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
    msg = "get probability table for source word to null"
    usage = "src2null_prob.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--source", type=str, required=True,
                        help="source corpus")
    parser.add_argument("--alignment", type=str, required=True,
                        help="alignment file")
    parser.add_argument("--output", type=str, help="output path")
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parseargs()
    source = open(args.source, 'r').read()
    lines_source = source.split('\n')
    if lines_source[-1].strip() == '':
        del lines_source[-1]

    align = open(args.alignment, 'r').read()
    lines_align = align.split('\n')
    if lines_align[-1].strip() == '':
        del lines_align[-1]

    # result: {'word':[null_count, total_count, prob]}
    result = {}

    for i in range(len(lines_source)):
        if (i+1) % 10000 == 0:
            print(i+1)
        s = lines_source[i]
        a = lines_align[i]
        words = s.split(' ')
        aligned = [0] * len(words)
        for tmp in a.split(' '):
            srcpos, trgpos = tmp.split('-')
            aligned[int(srcpos)] = 1
        for j in range(len(words)):
            if not result.has_key(words[j]):
                result[words[j]] = [0,0]
            result[words[j]][0] += aligned[j]
            result[words[j]][1] += 1

    result = {i: [result[i][0], result[i][1], 1-1.0*result[i][0]/result[i][1]] for i in result}
    
    json.dump(result, open(args.output, 'w'))

