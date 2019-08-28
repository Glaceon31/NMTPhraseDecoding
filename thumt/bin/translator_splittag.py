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


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
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
        segments_i = [k.strip() for k in segments_i]
        for j in range(len(segments_i)):
            if segments_i[j] == '':
                segments_i[j] = -1
            else:
                middle.write(segments_i[j]+'\n')
                segments_i[j] = linenum
                linenum += 1
        segments.append(segments_i)

    middle.close()
    ROOT='/data/zjctmp/AAAI2019_lexcons'
    CODE=ROOT+'/NMTPhraseDecoding'
    BIN=CODE+'/thumt/bin/translator.py'
    MODEL=ROOT+'/model_thumt'
    CMD='PYTHONPATH='+CODE+' CUDA_VISIBLE_DEVICES=5 python '+BIN+' --vocabulary '+MODEL+'/vocab.src.txt '+MODEL+'/vocab.trg.txt --model transformer --checkpoint '+MODEL+'/model.ckpt-43000 --input tmpprocess.in --output tmpprocess.out --parameters shared_embedding_and_softmax_weights=true,layer_preprocess=layer_norm,layer_postprocess=none'
    print(CMD)
    os.system(CMD)

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
    os.system('rm tmpprocess.out')
