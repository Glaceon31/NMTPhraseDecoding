from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import time
import math
import json

import copy

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
    force = {}
    for i in range(len(golden)):
        tmp = golden[i].split('\t')
        if len(tmp) == 2:
            key, value = tmp
            goldphrase[key] = value
        elif len(tmp) == 3:
            key, value, sen_num = tmp
            if force.has_key(int(sen_num)):
                force[int(sen_num)-1][key] = value
            else:
                force[int(sen_num)-1] = {key: value}
        elif len(tmp) == 4:
            key, value, sen_st, sen_ed = tmp
            for j in range(int(sen_st)-1, int(sen_ed)):
                if force.has_key(j):
                    force[j][key] = value
                else:
                    force[j] = {key: value}


    result = []
    result_sockeye = []
    num_constraint = 0
    for i in range(len(lines_input)):
        current_gold = copy.deepcopy(goldphrase)
        if force.has_key(i):
            for k in force[i].keys():
                current_gold[k] = force[i][k]
        found_gold = []
        sockeye_lexcons = []
        print(lines_input[i])
        words = lines_input[i].split(' ')
        pos = 0
        while pos < len(words): 
            for j in range(pos+1, len(words)+1):
                phrase = ' '.join(words[pos:j])
                if current_gold.has_key(phrase):
                    if j+1 >= len(words) or (j+1 < len(words) and not current_gold.has_key(' '.join(words[pos:(j+1)]))):
                        new_gold = [pos, j, phrase, current_gold[phrase]]
                        print('new gold:', pos, j)
                        found_gold.append(new_gold)
                        words[pos] = '<cons translation="'+current_gold[phrase]+'"> '+phrase+' </cons>'
                        sockeye_lexcons.append(current_gold[phrase])
                        for p in range(pos+1, j):
                            words[p] = ''
            pos += 1
        tmp = ' '.join(' '.join(words).split())
        result.append(tmp)
        tmp_sockeye = lines_input[i]
        for sk in sockeye_lexcons:
            num_constraint += 1
            tmp_sockeye += '\t'+sk
        result_sockeye.append(tmp_sockeye)

    print('num_constraints:', num_constraint)
    output = open(args.output, 'w')
    output.write('\n'.join(result))
    output.close()
    if args.sockeye:
        output = open(args.sockeye, 'w')
        output.write('\n'.join(result_sockeye))
        output.close()
