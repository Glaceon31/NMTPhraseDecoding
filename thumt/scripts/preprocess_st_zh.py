from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import numpy 
import time
import math
import json
import re
import thulac


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bpedir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    content = open(args.input, 'r').read()
    lines = content.split('\n')
    if lines[-1] == '':
        del lines[-1]

    middle = open('tmpprocess.in', 'w')
    tags = []
    segments = []
    linenum = 0
    for i in range(len(lines)):
        line = lines[i]
        tags_i = re.findall('<.*?>', line)
        tags.append(tags_i)
        tagre = re.compile('<.*?>')
        segments_i = tagre.split(line)
        for j in range(len(segments_i)):
            if segments_i[j] == '':
                segments_i[j] = -1
            else:
                middle.write(segments_i[j]+'\n')
                segments_i[j] = linenum
                linenum += 1
        segments.append(segments_i)

    middle.close()
    ## preprocess
    # normalize & tokenize
    os.system('python3 ~/utils/norm-char.py < tmpprocess.in > tmpprocess.norm')

    # segment
    thu1 = thulac.thulac(seg_only=True)
    thu1.cut_f('tmpprocess.norm', 'tmpprocess.thulac')

    # bpe
    os.system('python ~/utils/BPE/subword-nmt/apply_bpe.py --vocabulary '+args.bpedir+'/vocab.zh --vocabulary-threshold 50 -c '+args.bpedir+'/bpe32k < tmpprocess.thulac > tmpprocess.out' )
    middle_out = open('tmpprocess.out', 'r')

    lines_processed = middle_out.read().split('\n')
    output = open(args.output, 'w')
    for i in range(len(lines)):
        segments_i = segments[i]
        tags_i = tags[i]
        result = ''
        for j in range(len(segments_i)-1):
            if segments_i[j] != -1:
                result += ' '+lines_processed[segments_i[j]]
            result += ' '+tags_i[j]
        if segments_i[-1] != -1:
            result += ' '+lines_processed[segments_i[-1]]
        output.write(result.strip()+'\n')
    output.close()

    os.system('rm tmpprocess.in')
    os.system('rm tmpprocess.norm')
    os.system('rm tmpprocess.thulac')
    os.system('rm tmpprocess.out')

        
