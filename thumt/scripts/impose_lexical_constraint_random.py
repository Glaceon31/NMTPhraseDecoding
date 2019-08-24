from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import time
import math
import json
import random


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--alignment", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sockeye", type=str)
    parser.add_argument("--num", type=int, default=1)

    return parser.parse_args()

def continous(sorted_list, words):
    if len(sorted_list) == 0:
        return False
    for i in range(0, len(sorted_list)-1):
        if sorted_list[i+1]-sorted_list[i] != 1:
            return False
    if words[sorted_list[-1]].endswith('@@') or (sorted_list[0] > 0 and words[sorted_list[0]-1].endswith('@@')):
        return False
    return True

if __name__ == "__main__":
    args = parseargs()
    lines_input = open(args.input, 'r').read().split('\n')
    if lines_input[-1] == '':
        del lines_input[-1]
    reference = open(args.reference, 'r').read().split('\n')
    if reference[-1] == '':
        del reference[-1]
    alignment = open(args.alignment, 'r').read().split('\n')
    if alignment[-1] == '':
        del alignment[-1]

    result = []
    result_sockeye = []
    for i in range(len(lines_input)):
        found_gold = []
        sockeye_lexcons = []
        #print(lines_input[i])
        #print(reference[i])
        words = lines_input[i].split(' ')
        words_target = reference[i].split(' ')
        if reference[i] == '':
            continue
        valid = []
        constraint = []
        # parse alignment
        aligns = alignment[i].split(' ')
        t2s = []
        s2t = []
        for j in range(len(words_target)):
            t2s.append([])
            s2t.append([])
        for align in aligns:
            s, t = align.split('-')
            t2s[int(t)].append(int(s))
        # find valid positions
        pos = len(words_target)-1
        ss2tt = []
        tt = []
        while pos > 0: 
            tt.append(pos)
            if not words_target[pos-1].endswith('@@'):
                ss = sorted(list(set(t2s[pos])))
                if continous(ss, words):
                    ss2tt.append([ss, tt])
                tt = []
                pos -= 1
            else:
                t2s[pos-1] += t2s[pos]
                pos -= 1
        # random
        while len(constraint) < args.num and len(constraint) < len(ss2tt):
            ran = random.randint(0, len(ss2tt)-1)
            if not ran in constraint:
                constraint.append(ran)
        #print(ss2tt)
        for idx in constraint:
            start_s = ss2tt[idx][0][0]
            end_s = ss2tt[idx][0][-1]
            start_t = ss2tt[idx][1][-1]
            end_t = ss2tt[idx][1][0]
            target = ' '.join(words_target[start_t:end_t+1])
            words[start_s] = '<cons translation="'+target+'"> '+words[start_s]
            words[end_s] = words[end_s]+' </cons>'
            sockeye_lexcons.append(target)

        result.append(' '.join(words))
        tmp_sockeye = lines_input[i]
        for sk in sockeye_lexcons:
            tmp_sockeye += '\t'+sk
        result_sockeye.append(tmp_sockeye)
        #exit()


    output = open(args.output, 'w')
    output.write('\n'.join(result))
    output.close()
    if args.sockeye:
        output = open(args.sockeye, 'w')
        output.write('\n'.join(result_sockeye))
        output.close()
