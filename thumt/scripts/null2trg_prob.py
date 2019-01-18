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
    msg = "get probability table for null to target word"
    usage = "null2trg_prob.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--target", type=str, required=True,
                        help="target corpus")
    parser.add_argument("--alignments", type=str, nargs="+", required=True,
                        help="alignment files")
    parser.add_argument("--output", type=str, help="output path")
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parseargs()
    target = open(args.target, 'r').read()
    lines_target = target.split('\n')
    if lines_target[-1].strip() == '':
        del lines_target[-1]

    lines_align_list = []
    for i in range(len(args.alignments)):
        align = open(args.alignments[i], 'r').read()
        lines_align = align.split('\n')
        if lines_align[-1].strip() == '':
            del lines_align[-1]
        lines_align_list.append(lines_align)

    # result: {'word':[null_count, total_count, prob]}
    result = {}

    for senid in range(len(lines_target)):
        if (senid+1) % 10000 == 0:
            print(senid+1)
        for i in range(len(args.alignments)):
            sen = lines_target[senid]
            a = lines_align_list[i][senid]
            words = sen.split(' ')
            aligned = [0] * len(words)
            for tmp in a.split(' '):
                try:
                    srcpos, trgpos = tmp.split('-')
                except:
                    print('failed:', tmp)
                    continue
                aligned[int(trgpos)] = 1
            for j in range(len(words)):
                if not result.has_key(words[j]):
                    result[words[j]] = [0,0]
                result[words[j]][0] += aligned[j]
                result[words[j]][1] += 1

    result = {i: [result[i][0], result[i][1], 1-1.0*result[i][0]/result[i][1]] for i in result}
    
    json.dump(result, open(args.output, 'w'))

