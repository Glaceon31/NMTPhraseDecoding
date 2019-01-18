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
import math


def parseargs():
    msg = "Generate translation options"
    usage = "extract_translation_option.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--input", type=str, required=True,
                        help="origin phrase-table.gz from Moses")
    parser.add_argument("--num_options", type=int, default=10)
    parser.add_argument("--minprob", type=float, default=0.)
    parser.add_argument("--output", type=str, help="output path")
    parser.add_argument("--phrase", action="store_true", help="using high quality phrases")
    parser.add_argument("--prob", type=str, default="third", help="strategy for getprob")
    
    return parser.parse_args()


def isgood(probs, counts):
    ps = probs.split(' ')
    ps = [float(i) for i in ps]
    cs = counts.split(' ')
    cs = [float(i) for i in cs]
    if cs[2] < 5 or ps[0] < 0.01 or ps[2] < 0.1:
        return False
    return True


strategy = "third"
def get_prob(probs):
    global strategy
    if strategy == "first":
        return float(probs.split(' ')[0])
    elif strategy == "third":
        return float(probs.split(' ')[2])
    elif strategy == "product":
        return math.sqrt(float(probs.split(' ')[0])*float(probs.split(' ')[2]))
    elif strategy == "product_all":
        return math.pow(float(probs.split(' ')[0])*float(probs.split(' ')[1])*float(probs.split(' ')[2])*float(probs.split(' ')[3]), 0.25)
    elif strategy == "product_all_2":
        return math.pow(float(probs.split(' ')[0])*pow(float(probs.split(' ')[1]), 2)*float(probs.split(' ')[2])*float(probs.split(' ')[3]), 0.2)
    elif strategy == "average":
        return sum([float(i) for i in probs.split(' ')])/4


def get_options(trg_list, args):
    result = sorted(trg_list, key=lambda x: get_prob(x[1]), reverse=True)
    result = result[:args.num_options]
    result = [[i[0], get_prob(i[1])] for i in result]
    return result


def rbpe(inp):
    return inp.replace('@@ ', '')


if __name__ == "__main__":
    global strategy
    args = parseargs()
    strategy = args.prob
    content = open(args.input, 'r')
    result = {}
    count = 0
    findmax = False
    current_src = ''
    current_trg = []
    num = 0 
    phrasenum = 0
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
                if get_prob(probs) > args.minprob:
                    current_trg.append([trg, probs])
        if len(rbpe(src).split(' ')) == 1 and not findmax:
            current_src = src
            findmax = True
            current_trg.append([trg, probs])
        if args.phrase:
            if len(rbpe(src).split(' ')) > 1 :
                if isgood(probs, counts):
                    if result.has_key(src):
                        result[src].append([trg, get_prob(probs)])
                    else:
                        result[src] = [[trg, get_prob(probs)]]
                        phrasenum += 1
        if count % 100000 == 0:
            print(num, '&', phrasenum , '/', count)
        line = content.readline()
    json.dump(result, open(args.output, 'w'))


