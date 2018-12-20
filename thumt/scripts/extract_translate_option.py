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
    parser.add_argument("--num_options", type=int, default=10)
    parser.add_argument("--output", type=str, help="output path")
    
    return parser.parse_args()


def get_options(trg_list, args):
    result = sorted(trg_list, key=lambda x: float(x[1].split(' ')[2]), reverse=True)
    result = result[:args.num_options]
    result = [[i[0], float(i[1].split(' ')[2])] for i in result]
    return result


def rbpe(inp):
    return inp.replace('@@ ', '')


if __name__ == "__main__":
    args = parseargs()
    content = open(args.input, 'r')
    result = {}
    count = 0
    findmax = False
    current_src = ''
    current_trg = []
    num = 0 
    line = content.readline()
    while line:
        line = line.strip()
        count += 1
        try:
            src, trg, probs, align, counts, _ = line.split(' ||| ')
        except:
            print('line', count, 'bad')
            print(line)
            count += 1
            continue
        if findmax:
            if src != current_src: 
                result[current_src] = get_options(current_trg, args)
                num += 1
                findmax = False
                current_trg = []
            else:
                current_trg.append([trg, probs])
        if len(rbpe(src).split(' ')) == 1 and not findmax:
            current_src = src
            findmax = True
            current_trg.append([trg, probs])
        if count % 100000 == 0:
            print(num, '/', count)
        line = content.readline()
    json.dump(result, open(args.output, 'w'))


