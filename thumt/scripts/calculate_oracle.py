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
    parser.add_argument("--output", type=str, required=True,
                        help="output")
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


def bleu_appro(hypo_c, refs_dict, n, hypo_dict=None):
    '''
        hypo: string, one sentence per line
        refs: [string, one sentence per line]
        n: int
    '''
    correctgram_count = [0]*n
    ngram_count = [0]*n
    hypo_sen = hypo_c.split('\n')
    #refs_sen = [refs_c[i].split('\n') for i in range(len(refs_c))]
    for num in range(len(hypo_sen)):
        hypo = hypo_sen[num]

        #refs = [refs_sen[i][num] for i in range(len(refs_c))]
        #refs_dict = {}
        #for i in range(len(refs)):
        #    ref = refs[i]
        #    ref_dict = sentence2dict(ref, n)
        #    refs_dict = merge_dict(refs_dict, ref_dict)

        if not hypo_dict:
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


def bleu(hypo_c, refs_c, n, refs_dict=None, hypo_dict=None):
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
        if not refs_dict:
            refs_dict = {}
            for i in range(len(refs)):
                ref = refs[i]
                ref_dict = sentence2dict(ref, n)
                refs_dict = merge_dict(refs_dict, ref_dict)
        ref_lengths = sorted([len(refs[i].split(' ')) for i in range(len(refs))])
        ref_distances = [abs(r-h_length) for r in ref_lengths]

        ref_length += ref_lengths[numpy.argmin(ref_distances)]

    bp = 1
    if hypo_length < ref_length:
        bp = math.exp(1-ref_length*1.0/hypo_length)
    return bp * bleu_appro(hypo_c, refs_dict, n, hypo_dict=hypo_dict)


def num_ngram_match(hypo_c, refs_c, n, refs_dict=None):
    hypo_dict = sentence2dict(hypo_c, n)
    correctgram_count = [0]*n
    ngram_count = [0]*n
    for key in hypo_dict:
        value = hypo_dict[key]
        length = len(key.split(' '))
        #print key,length
        ngram_count[length-1] += value
        if refs_dict.has_key(key):
            correctgram_count[length-1] += min(value, refs_dict[key])

    return sum(correctgram_count)


def remove_duplicate(inp):
    result = []
    for i in inp:
        if not i in result:
            result.append(i)
    return result


def insert(previous, new, refs, ngram, refs_dict=None):
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
        tmp = ' '.join(words[:i])+' '+new+' '+' '.join(words[i+1:])
        tmp = tmp.strip()
        ev = num_ngram_match
        b = ev(tmp, refs, ngram, refs_dict=refs_dict)
        if b > max:
            max = b
            result = [tmp, b]
    return result


def oracle(src, refs, phrases_all, ngram, beam_size): 
    src_words = src.decode('utf-8').split(' ')
    phrase = subset(phrases_all, src_words, ngram)
    #print('phrase', phrase)
    stacks = [[['', 0]]]

    refs_dict = {}
    for i in range(len(refs)):
        ref = refs[i]
        ref_dict = sentence2dict(ref, ngram)
        refs_dict = merge_dict(refs_dict, ref_dict)

    for i in range(len(src_words)):
        #print(i)
        current = []
        for start in range(max(i+1-ngram, 0), i+1):
            src = ' '.join(src_words[start:i+1])
            if phrase.has_key(src):
                #print('src_word', src.encode('utf-8'))
                #print('pharse_word:', [t.encode('utf-8') for t in phrase[src]])
                for pre in stacks[start]:
                    for p in phrase[src]:
                        newstat = insert(pre[0], p, refs, ngram, refs_dict=refs_dict)
                        if not newstat == '':
                            current.append(newstat)
        current = remove_duplicate(current)
        current = sorted(current, key=lambda x:x[1], reverse=True)
        current = current[:beam_size]
        if len(current) == 0:
            current = [['', 0]]
        #print("current", current)
        stacks.append(current)
    return current[0][0]


if __name__ == "__main__":
    args = parseargs()

    src = open(args.source, 'r').read() 
    refs = [open(ref, 'r').read() for ref in args.reference]

    src_sens = src.split('\n')
    if src_sens[-1] == '':
        del src_sens[-1]
    ref_sens = [r.split('\n') for r in refs]
    # Load phrase table
    phrase_table = json.load(open(args.phrase, 'r'))

    hypos = []
    out = open(args.output, 'w')
    for num in range(len(src_sens)):
        print(src_sens[num])
        refs = [ref_sens[i][num] for i in range(len(ref_sens))]
        new_hypo = oracle(src_sens[num], refs, phrase_table, 4, 4)
        hypos.append(new_hypo)
        print(new_hypo.encode('utf-8'))
        print(bleu(new_hypo, refs, 4))
        out.write(new_hypo.encode('utf-8')+'\n')
    print(bleu('\n'.join(hypos), refs, 4))
