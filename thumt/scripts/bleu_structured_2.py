from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os
import re

import numpy 
import time
import math
import json


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hypo", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--bleu", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tag", action="store_true")

    return parser.parse_args()


def get_tag_name(word):
    if is_start_tag(word):
        return word[1:-1].split(' ')[0]
    elif is_end_tag(word):
        return word[2:-1].split(' ')[0]
    assert 0 == 1 


def is_end_tag(word):
    if word[0] == '<' and word[1] == '/' and word[-1] == '>':
        return True
    return False


def is_start_tag(word):
    if word[0] == '<' and word[1] != '/' and word[-1] == '>':
        return True
    return False


def is_tag(word):
    if is_start_tag(word) or is_end_tag(word):
        return True
    return False


def splitline(line, args):
    result = line.split()
    pos = 0
    while pos < len(result):
        if result[pos][0] == '<' and result[pos][-1] != '>':
            if pos+1 < len(result):
                result[pos] += ''+result[pos+1]
                del result[pos+1]
            else:
                print('wrong!! ', result[pos])
                exit()
        else:
            pos += 1
    return result

def removetitle(line):
    result = re.sub('title=".*?"', '', line)
    return result


if __name__ == "__main__":
    args = parseargs()
    hypo = open(args.hypo, 'r').read()
    lines_hypo = hypo.split('\n')
    if lines_hypo[-1] == '':
        del lines_hypo[-1]
    
    ref = open(args.ref, 'r').read()
    lines_ref = ref.split('\n')
    if lines_ref[-1] == '':
        del lines_ref[-1]

    
    middle_hypo = open('tmp.hypo', 'w')
    middle_ref = open('tmp.ref', 'w')

    for i in range(len(lines_hypo)):
        line_hypo = lines_hypo[i]
        line_ref = lines_ref[i]
        words_hypo = splitline(line_hypo, args)
        words_ref = splitline(line_ref, args)
        words_hypo = [removetitle(w) for w in words_hypo]
        words_ref = [removetitle(w) for w in words_ref]
        middle_hypo.write(' '.join(words_hypo)+'\n')
        middle_ref.write(' '.join(words_ref)+'\n')

    middle_hypo.close()
    middle_ref.close()
    os.system("perl "+args.bleu+" -lc tmp.ref < tmp.hypo > " +args.output)

    os.system("rm tmp.hypo")
    os.system("rm tmp.ref")

