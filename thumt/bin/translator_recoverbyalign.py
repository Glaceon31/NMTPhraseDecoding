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

    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--trg", type=str, required=True)
    parser.add_argument("--align", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()

def splitline(line):
    result = line.split(' ')
    pos = 0
    while pos < len(result):
        if result[pos][0] == '<' and result[pos][-1] != '>':
            result[pos] += ' '+result[pos+1]
            del result[pos+1]
        else:
            pos += 1
    return result

def reverse_bpe(inp):
    return inp.replace('@@ ', '')

def isstarttag(word):
    if word[0] == '<' and word[1] != '/' and word[-1] == '>':
        return True
    return False

def gettagname(word):
    tagcontent = re.findall('<(.*?)>', word)[0]
    return tagcontent.split(' ')[0]

def isendtag(word, tagname=None):
    if word[0] == '<' and word[1] == '/' and word[-1] == '>':
        endtagcontent = re.findall('</(.*?)>', word)[0]
        endtagname = endtagcontent.split(' ')[0]
        if not tagname or endtagname == tagname:
            return True
    return False

def getmatchscore(words, translated):
    result = 0
    for word in words:
        if word in translated:
            result += 1
        else:
            return -1
    return result


def parse_align(align):
    result = []
    for i in range(200):
        result.append([])
    for align_w in align.split(' '):
        pos_src, pos_trg = align_w.split('-')
        pos_src = int(pos_src)
        pos_trg = int(pos_trg)
        result[pos_src].append(pos_trg)
    return result


if __name__ == "__main__":
    args = parseargs()
    content = open(args.src, 'r').read()
    lines = content.split('\n')
    if lines[-1] == '':
        del lines[-1]

    alignf = open(args.align, 'r').read()
    lines_align = alignf.split('\n')
    if lines_align[-1] == '':
        del lines_align[-1]

    content_raw = open(args.trg, 'r').read()
    lines_raw = content_raw.split('\n')
    if lines_raw[-1] == '':
        del lines_raw[-1]

    middle = open('tmpprocess.in', 'w')
    linenum = 0

    outf = open(args.output, 'w')
    for i in range(len(lines)):
        print("=== %d ===" % i)
        line = lines[i]
        words = splitline(line)
        tags_i = []
        start_tag = [] #[[start_raw_pos, start_name, end_raw_pos, end_name]]
        pos_raw = 0
        for pos in range(len(words)):
            level = 0
            if isstarttag(words[pos]):
                start_tag.append([pos_raw, words[pos]])
            elif isendtag(words[pos]):
                assert len(start_tag) > 0
                assert isendtag(words[pos], gettagname(start_tag[-1][1]))
                tags_i.append(start_tag[-1]+[pos_raw, words[pos]])
                del start_tag[-1]
            else:
                pos_raw += 1

        alignment = parse_align(lines_align[i])
        words_raw = lines_raw[i].split(' ')
        for tag in tags_i:
            aligned = []
            pos_s, tag_s, pos_e, tag_e = tag
            for j in range(pos_s, pos_e):
                if alignment[j] != -1:
                    aligned += alignment[j]
            if len(aligned) == 0:
                continue
            aligned = sorted(aligned)
            #cs, ce = max_continous(aligned)
            cs = aligned[0]
            ce = aligned[-1]
            while words_raw[ce].endswith('@@'):
                ce += 1
            while cs > 0 and words_raw[cs-1].endswith('@@'):
                cs -= 1
            words_raw[cs] = tag_s+' '+words_raw[cs]
            words_raw[ce] = words_raw[ce]+' '+tag_e
        outf.write(' '.join(words_raw)+'\n')
                
