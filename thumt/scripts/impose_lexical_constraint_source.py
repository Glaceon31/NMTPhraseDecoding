from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import time
import math
import json


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--golden", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sockeye", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    lines_input = open(args.input, 'r').read().split('\n')
    if lines_input[-1] == '':
        del lines_input[-1]
    golden = open(args.golden, 'r').read().split('\n')
    if golden[-1] == '':
        del golden[-1]
    goldphrase = {}
    for i in range(len(golden)):
        key, value = golden[i].split('\t')
        goldphrase[key] = value

    result = []
    result_sockeye = []
    for i in range(len(lines_input)):
        found_gold = []
        sockeye_lexcons = []
        print(lines_input[i])
        words = lines_input[i].split(' ')
        pos = 0
        while pos < len(words): 
            for j in range(pos+1, len(words)+1):
                phrase = ' '.join(words[pos:j])
                if goldphrase.has_key(phrase):
                    if j+1 >= len(words) or (j+1 < len(words) and not goldphrase.has_key(' '.join(words[pos:(j+1)]))):
                        new_gold = [pos, j, phrase, goldphrase[phrase]]
                        print('new gold:', pos, j)
                        found_gold.append(new_gold)
                        words[pos] = '<cons translation="'+goldphrase[phrase]+'"> '+phrase+' </cons>'
                        sockeye_lexcons.append(goldphrase[phrase])
                        for p in range(pos+1, j):
                            words[p] = ''
            pos += 1
        tmp = ' '.join(' '.join(words).split())
        result.append(tmp)
        tmp_sockeye = lines_input[i]
        for sk in sockeye_lexcons:
            tmp_sockeye += '\t'+sk
        result_sockeye.append(tmp_sockeye)

    output = open(args.output, 'w')
    output.write('\n'.join(result))
    output.close()
    if args.sockeye:
        output = open(args.sockeye, 'w')
        output.write('\n'.join(result_sockeye))
        output.close()
