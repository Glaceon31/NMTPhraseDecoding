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

def isstarttag(word):
    if word[0] == '<' and word[1] != '/' and word[-1] == '>':
        return True
    return False

def gettagname(word):
    tagcontent = re.findall('<(.*?)>', word)[0]
    return tagcontent.split(' ')[0]

def isendtag(word, tagname):
    if word[0] == '<' and word[1] == '/' and word[-1] == '>':
        endtagcontent = re.findall('</(.*?)>', word)[0]
        endtagname = endtagcontent.split(' ')[0]
        if endtagname == tagname:
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


if __name__ == "__main__":
    args = parseargs()
    content = open(args.input, 'r').read()
    lines = content.split('\n')
    if lines[-1] == '':
        del lines[-1]

    middle = open('tmpprocess.in', 'w')
    tags = []
    sentences = []
    segments = []
    linenum = 0
    for i in range(len(lines)):
        line = lines[i]
        sentence = re.sub('<.*?>', '', line)
        sentences.append(linenum)
        linenum += 1
        middle.write(sentence.strip()+'\n')
        words = splitline(line)
        tags_i = []
        for pos in range(len(words)):
            if isstarttag(words[pos]):
                tagname = gettagname(words[pos])
                pos_end = -1
                for tmp in range(pos+1, len(words)):
                    if isendtag(words[tmp], tagname):
                        pos_end = tmp
                        break
                if pos_end != -1:
                    middle.write(re.sub('<.*?>', '', ' '.join(words[pos+1:tmp])).strip()+'\n')
                    newtag = [words[pos], words[tmp], ' '.join(words[pos+1:tmp]), linenum]
                    tags_i.append(newtag)
                    linenum += 1
        tags.append(tags_i)
                
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
        sentences_i = sentences[i]
        tags_i = tags[i]
        result = lines_processed[sentences_i]
        words_i = result.split(' ')
        words_result = result.split(' ')
        '''
        for j in range(len(tags_i)):
            translated = lines_processed[tags_i[j][3]]
            words_translated = translated.split(' ')
            maxscore = 0
            maxresult = -1
            for tmpst in range(len(words_i)):
                for tmped in range(tmpst, len(words_i)):
                    tmpscore = getmatchscore(words_i[tmpst:tmped+1], words_translated)
                    if tmpscore == -1:
                        break
                    if tmpscore > maxscore:
                        maxscore = tmpscore
                        maxresult = [tmpst, tmped]

            if maxresult != -1:
                words_result[maxresult[0]] = tags_i[j][0]+' '+words_result[maxresult[0]]
                words_result[maxresult[1]] += ' '+tags_i[j][1]
        '''

        output.write(' '.join(words_result).strip()+'\n')


    output.close()
    print(getmatchscore(['gas','explosion','in','20@@','13'],'ra@@ z@@ ali gas explosion in 20@@ 13'.split(' ')))

    #os.system('rm tmpprocess.in')
    #os.system('rm tmpprocess.out')
