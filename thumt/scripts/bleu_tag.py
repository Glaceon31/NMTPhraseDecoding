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

from calculate_oracle import bleu


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
        ref_segs = []
        ref_tags = []
        for pos in range(len(words_ref)):
            if is_start_tag(words_ref[pos]):
                pos_end = -1
                level = 0
                tagname = get_tag_name(words_ref[pos])
                for tmp in range(pos+1, len(words_ref)):
                    if is_start_tag(words_ref[tmp]):
                        level += 1
                    if is_end_tag(words_ref[tmp]):
                        if level > 0:
                            level -= 1
                        else:
                            pos_end = tmp
                            break
                if pos_end != 1:
                    ref_segs.append(re.sub('<.*?>', '', ' '.join(words_ref[pos+1:tmp])).strip())
                    ref_tags.append(tagname)
        hypo_segs = ['']*len(ref_segs)
        for pos in range(len(words_hypo)):
            if is_start_tag(words_hypo[pos]):
                pos_end = -1
                level = 0
                tagname = get_tag_name(words_hypo[pos])
                for tmp in range(pos+1, len(words_hypo)):
                    if is_start_tag(words_hypo[tmp]):
                        level += 1
                    if is_end_tag(words_hypo[tmp]):
                        if level > 0:
                            level -= 1
                        else:
                            pos_end = tmp
                            break
                hypo_seg = re.sub('<.*?>', '', ' '.join(words_hypo[pos+1:tmp])).strip()
                maxbleu = 0.
                maxj = -1
                for j in range(len(ref_segs)):
                    if hypo_segs[j] == '' and tagname == ref_tags[j]:
                        tmpbleu = bleu(hypo_seg, [ref_segs[j]], 4)
                        if tmpbleu > maxbleu:
                            maxbleu = tmpbleu
                            maxj = j
                if maxj != -1:
                    hypo_segs[maxj] = hypo_seg
                    
        if len(ref_segs) > 0:
            middle_hypo.write('\n'.join(hypo_segs)+'\n')
            middle_ref.write('\n'.join(ref_segs)+'\n')

    middle_hypo.close()
    middle_ref.close()
    os.system("perl "+args.bleu+" -lc tmp.ref < tmp.hypo > " +args.output)

    #os.system("rm tmp.hypo")
    #os.system("rm tmp.ref")

