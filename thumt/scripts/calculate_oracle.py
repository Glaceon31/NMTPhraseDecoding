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


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--phrase", type=str, required=True,
                        help="phrase table")
    parser.add_argument("--source", type=str, required=True,
                        help="source")
    parser.add_argument("--reference", type=str, required=True, nargs="+",
                        help="references")

    return parser.parse_args()


def subset(phrases, words, ngram):
    result = {}
    covered = [0] * len(words)
    for i in range(len(words)):
        for j in range(i+1, min(i+ngram+1, len(words)+1)):
            phrase = ' '.join(words[i:j])
            if phrases.has_key(phrase):
                result[phrase] = phrases[phrase]
                for k in range(i, j):
                    covered[k] = 1
    # special treatment for words with no phrase
    for i in range(len(words)):
        if result.has_key(words[i]):
            result[words[i]].append(words[i])
        else:
            result[words[i]] = [words[i]]
            
    return result


def merge_dict(d1,d2):
    result = d1
    for key in d2:
        value = d2[key]
        if result.has_key(key):
            result[key] = max(result[key],value)
        else:
            result[key] = value
    return result


def sentence2dict(sentence, n):
    words = sentence.split(' ')
    result = {}
    for n in range(1,n+1):
        for pos in range(len(words)-n+1):
            gram = ' '.join(words[pos:pos+n])
            if result.has_key(gram):
                result[gram] += 1
            else:
                result[gram] = 1
    return result


def bleu_appro(hypo_c, refs_dict, n):
    '''
        hypo: string, one sentence per line
        refs: [string, one sentence per line]
        n: int
    '''
    correctgram_count = [0]*n
    ngram_count = [0]*n
    hypo_sen = hypo_c.split('\n')
    refs_sen = [refs_c[i].split('\n') for i in range(len(refs_c))]
    for num in range(len(hypo_sen)):
        hypo = hypo_sen[num]

        refs = [refs_sen[i][num] for i in range(len(refs_c))]
        refs_dict = {}
        for i in range(len(refs)):
            ref = refs[i]
            ref_dict = sentence2dict(ref, n)
            refs_dict = merge_dict(refs_dict, ref_dict)

        hypo_dict = sentence2dict(hypo, n)
        for key in hypo_dict:
            value = hypo_dict[key]
            length = len(key.split(' '))
            #print key,length
            ngram_count[length-1] += value
            if refs_dict.has_key(key):
                correctgram_count[length-1] += min(value, refs_dict[key])

    result = 0.
    bleu_n = [0.]*n
    if correctgram_count[0] == 0:
        return 0.
    for i in range(n):
        if correctgram_count[i] == 0:
            correctgram_count[i] += 1
            ngram_count[i] += 1
        bleu_n[i] = correctgram_count[i]*1./ngram_count[i]
        result += math.log(bleu_n[i])/n

    return math.exp(result)


def bleu(hypo_c, refs_c, n):
    '''
        hypo: string, one sentence per line
        refs: [string, one sentence per line]
        n: int
    '''
    hypo_sen = hypo_c.split('\n')
    refs_sen = [refs_c[i].split('\n') for i in range(len(refs_c))]
    hypo_length = 0
    ref_length = 0
    for num in range(len(hypo_sen)):
        hypo = hypo_sen[num]
        h_length = len(hypo.split(' '))
        hypo_length += h_length
        
        refs = [refs_sen[i][num] for i in range(len(refs_c))]
        ref_lengths = sorted([len(refs[i].split(' ')) for i in range(len(refs))])
        ref_distances = [abs(r-h_length) for r in ref_lengths]

        ref_length += ref_lengths[numpy.argmin(ref_distances)]

    bp = 1
    if hypo_length < ref_length:
        bp = math.exp(1-ref_length*1.0/hypo_length)
    return bp * bleu_appro(hypo_c, refs_c, n)


def 


def insert(previous, new, refs, ngram):
    '''
        insert new phrase to partial translation where BLEU is maximized
        previous: partial translation
        new: added phrase
        refs: [reference sentence]
    '''
    words = previous.split(' ')
    max = 0
    result = ''
    for i in range(0, len(words)+1):
        tmp = ' '.join(words[:i])+' '+' '.join(words[i+1:])
        b = bleu(tmp, refs, ngram)
        if b > max:
            max = b
            result = tmp
    return result


def oracle(src, refs, phrases_all, ngram, beam_size): 
    src_words = src.decode('utf-8').split(' ')
    phrase = subset(phrases_all, src_words, ngram)
    print(phrase)
    stacks = [['', 0]]
    for i in range(len(src_words)):
        current = []
        for start in range(max(i-ngram, 0), i):
            src = ' '.join(src_words[start:i])
            if phrase.has_key(src):
                for pre in stacks[start]:
                    for p in phrase[src]:
                        current.append(insert(pre, p, refs, evaluation, ngram))
        current = [bleu(i, refs, ngram) for i in current]
        print(current)
        exit()

            
        


if __name__ == "__main__":
    args = parseargs()

    src = open(args.source, 'r').read() 
    refs = [open(ref, 'r').read() for ref in args.reference]

    src_sens = src.split('\n')
    ref_sens = [r.split('\n') for r in refs]
    # Load phrase table
    phrase_table = json.load(open(args.phrase, 'r'))

    hypos = []
    for num in range(len(src_sens)):
        hypos.append(oracle(src_sens[num], [ref_sens[i][num] for i in range(len(ref_sens))], phrase_table, 4, 4))
    print(bleu('\n'.join(hypos), refs, 4))
