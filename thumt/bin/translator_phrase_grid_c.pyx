#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import json
import copy

import tensorflow as tf
import thumt.data.vocab as vocabulary
import thumt.models as models

import numpy as np
import math
import time

cimport cython
from libc.stdlib cimport malloc, free
from cpython cimport array
import array
import pyximport
pyximport.install()
import thumt.utils.automatons_c as automatons_o
from libc.stdio cimport printf
from libc.string cimport strcpy, strcat, strlen, strcmp, strncpy
from libc.math cimport pow, log


cdef struct automatons_state:
    int next_state
    int *visible
    int num_visible


cdef struct automatons:
    int state_num
    automatons_state states[10]


cdef automatons automatons_build(words_src, punc, params):
    cdef automatons result
    cdef int length = len(words_src)
    cdef int start, i, j
    if params.punc_border:
        start = 0
        for i in range(length):
            if words_src[i] in punc:
                result.states[result.state_num].num_visible = i+1-start
                result.states[result.state_num].visible = <int *> malloc(result.states[result.state_num].num_visible*sizeof(int))
                for j in range(0, i+1-start):
                    result.states[result.state_num].visible[j] = j+start
                result.states[result.state_num].next_state = -1
                result.state_num += 1
                start = i+1
        if start < length:
            result.states[result.state_num].num_visible = length-start
            result.states[result.state_num].visible = <int *> malloc(result.states[result.state_num].num_visible*sizeof(int))
            for j in range(0, length-start):
                result.states[result.state_num].visible[j] = j+start
            result.states[result.state_num].next_state = -1
            result.state_num += 1
        for i in range(result.state_num-1):
            result.states[i].next_state = i+1
    else:
        result.states[0].num_visible = length
        result.states[0].visible = <int *> malloc(length*sizeof(int))
        for i in range(0, length):
            result.states[0].visible[i] = i
        result.states[0].next_state = -1
        result.state_num = 1
    return result


cdef print_autom(automatons autom):
    cdef int i, j
    for i in range(autom.state_num):
        printf("state %d:", i)
        printf("visible {")
        for j in range(autom.states[i].num_visible):
            printf("%d ", autom.states[i].visible[j])
        printf("} next_state %d\n", autom.states[i].next_state)


cdef int can_go_next(automatons_state state, int *coverage):
    cdef int i
    for i in range(state.num_visible):
        if coverage[state.visible[i]] == 0:
            return 0
    return 1


cdef struct translation_status:
    char *translation
    int *coverage
    float align_loss
    float src2null_loss
    float translation_loss
    int automatons
    int limited
    char *limits
    int hidden_state_id
    int previous[3]
    float loss


cdef translation_status copy_translation_status(translation_status old, int len_src):
    cdef translation_status new
    new = old
    cdef int *coverage
    coverage = <int *> malloc(len_src * sizeof(int))
    cdef int i 
    for i in range(len_src):
        coverage[i] = old.coverage[i]
    new.coverage = coverage
    return new


cdef struct beam:
    translation_status *content
    int count


cdef print_cand(candidate cand):
    printf("%s;%d %d;%f %f\n",cand.phrase, cand.pos, cand.pos_end, cand.loss, cand.prob_align)


cdef struct candidate:
    char *phrase
    int pos, pos_end
    float loss, prob_align 


cdef struct phrase_pair:
    char *phrase
    float prob


cdef phrase_pair* build_phrases_c(phrases):
    cdef phrase_pair* result = <phrase_pair*> malloc(500*sizeof(phrase_pair))
     
    return result


cdef struct loss_pair:
    char *translation
    float loss


cdef candidate copy_candidate(candidate cand):
    cdef candidate result
    #result.phrase = <char*> malloc(strlen(cand.phrase)*sizeof(char))
    #strcpy(result.phrase, cand.phrase)
    result.phrase = cand.phrase
    result.pos = cand.pos
    result.pos_end = cand.pos_end
    result.loss = cand.loss
    result.prob_align = cand.prob_align
    return result


cdef struct params_s:
    int beam_size, keep_status_num, split_limited, src2null_loss, bpe_phrase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True, 
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--phrase", type=str, required=True,
                        help="Name of the phrase table")
    parser.add_argument("--nullprob", type=str, required=True,
                        help="probability for source word to null")
    parser.add_argument("--null2trg", type=str,
                        help="vocabulary for null to target word")
    parser.add_argument("--stoplist", type=str,
                        help="stopword list banned from generation")
    parser.add_argument("--goldphrase", type=str,
                        help="golden phrase table")
    parser.add_argument("--ngram", type=int, default=4,
                        help="ngram length")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    parser.add_argument("--tmpphrase", type=str, default="",
                        help="")
    parser.add_argument("--rbpe", action="store_true", 
                        help="")
    parser.add_argument("--verbose", action="store_true", 
                        help="")
    parser.add_argument("--time", action="store_true", 
                        help="")



    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        model=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        device_list=[0],
        num_threads=6,
        eval_batch_size=32,
        # decoding
        beam_size=4,
        decode_alpha=0.6,
        decode_length=20,
        # phrase specific
        bpe_phrase=True,
        merge_status="max_align",
        keep_status_num=1,
        src2null_loss=1,
        split_limited=0,
        allow_src2stop=1,
        use_golden=1,
        punc_border=0,
        cut_ending=0,
        cut_threshold=4.
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def read_files(names):
    inputs = [[] for _ in range(len(names))]
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]

        for i, line in enumerate(lines):
            inputs[i].append(line)

        count += 1

    # Close files
    for fd in files:
        fd.close()

    return inputs


cdef get_feature_map(translation_status stack[100][4], int *stack_count, translation_status *stack_limit[100], int *stack_limit_count, int len_src, ivocab_trg, hidden_state_pool, params, int maps[100][4], int maps_limit[100][10000]):
    #printf('get_feature_map\n')
    features = {}
    sentence_list = {}
    cdef int sentence_num = 0
    new_hidden_pool = []
    cdef int num_cov, pos, idx, i, max_len
    cdef int count = 0
    cdef translation_status element
    for num_cov in range(len_src+1):
        #maps.append([])
        for idx in range(stack_count[num_cov]):
            count += 1
            element = stack[num_cov][idx]
            found = 0
            if sentence_list.has_key(element.translation):
                maps[num_cov][idx] = sentence_list[element.translation]
            else:
                sentence_list[element.translation] = sentence_num
                maps[num_cov][idx] = sentence_num
                sentence_num += 1
                new_hidden_pool.append(hidden_state_pool[element.hidden_state_id])
        for idx in range(stack_limit_count[num_cov]):
            count += 1
            element = stack_limit[num_cov][idx]
            found = 0
            if sentence_list.has_key(element.translation):
                maps_limit[num_cov][idx] = sentence_list[element.translation]
            else:
                sentence_list[element.translation] = sentence_num
                maps_limit[num_cov][idx] = sentence_num
                sentence_num += 1
                new_hidden_pool.append(hidden_state_pool[element.hidden_state_id])
            '''
            for i in range(len(sentence_list)):
                if element.translation == sentence_list[i]:
                    found = 1
                    pos = i
                    break
            if found == 1:
                maps[num_cov][idx] = pos
            else:
                sentence_list.append(element.translation)
                maps[num_cov][idx] = len(sentence_list)-1
                new_hidden_pool.append(hidden_state_pool[element.hidden_state_id])
            '''
    #printf('get_feature_map 0.5')
    '''
    for num_cov in range(len_src+1):
        #printf('get_feature_map 0.6: %d/%d\n', num_cov, stack_limit_count[num_cov])
        #maps_limit.append([])
        for idx in range(stack_limit_count[num_cov]):
            #printf('get_feature_map limit %d/%d\n', num_cov, idx)
            element = stack_limit[num_cov][idx]
            found = 0
            for i in range(len(sentence_list)):
                if element.translation == sentence_list[i]:
                    found = 1
                    pos = i
                    break
            if found == 1:
                maps_limit[num_cov][idx] = pos
            else:
                sentence_list.append(element.translation)
                maps_limit[num_cov][idx] = len(sentence_list)-1
                new_hidden_pool.append(hidden_state_pool[element.hidden_state_id])
    '''
    sen_ids_list = [0] * sentence_num
    for sen in sentence_list.keys():
        assert sen_ids_list[sentence_list[sen]] == 0
        sen_ids_list[sentence_list[sen]] = getid(ivocab_trg, sen)

    #sen_ids_list = [getid(ivocab_trg, sentence) for sentence in sentence_list]
    #num_sent = len(sen_ids_list)
    max_len = max(map(len, sen_ids_list))
    padded_input = np.ones([sentence_num, max_len], dtype=np.int32) * ivocab_trg['<pad>']
    for i in range(sentence_num):
        padded_input[i][:len(sen_ids_list[i])] = sen_ids_list[i]
    features["target"] = padded_input
    features["target_length"] = [len(sen_ids) for sen_ids in sen_ids_list]
    features["decoder"] = {}
    for i in range(params.num_decoder_layers):
        # the main timecost is here
        features["decoder"]["layer_%d" % i] = merge_tensor(new_hidden_pool, i)
    #print('maps:', maps)
    #print('maps_limit:', maps_limit)
    return features, len(sentence_list)


def get_feature(sentence_list, ivocab_trg):
    features = {}
    sen_ids_list = [getid(ivocab_trg, sentence[0]) for sentence in sentence_list]
    num_sent = len(sen_ids_list)
    max_len = max(map(len, sen_ids_list))
    padded_input = np.ones([num_sent, max_len], dtype=np.int32) * ivocab_trg['<pad>']
    for i in range(num_sent):
        padded_input[i][:len(sen_ids_list[i])] = sen_ids_list[i]
    features["target"] = padded_input
    features["target_length"] = [len(sen_ids) for sen_ids in sen_ids_list]
    return features


def reverse_phrase(phrases):
    result = {}
    for k in phrases.keys():
        for p in phrases[k]:
            if result.has_key(p[0]):
                result[p[0]].append(k)
            else:
                result[p[0]] = [k]
    return result


def filter_stop(phrases, stopword_list):
    result = []
    for p in phrases:
        stop = True
        for word in p[0].split(' '):
            if not word in stopword_list:
                stop = False
                break
        if not stop:
            result.append(p)
    return result


def transform_gold(phrase):
    words = phrase.split(' ')
    for i in range(len(words)-1):
        if not words[i].endswith('@@'):
            words[i] = words[i] + '@@'
    return ' '.join(words)


cdef subset(phrases, words, ngram, params, rbpe=False, stopword_list=None, goldphrase=None):
    result = {}
    words_result = copy.deepcopy(words)
    covered = [0] * len(words)
    golden = [0] * len(words)
    found_gold = [] # [start, end, src, trg]
    if params.use_golden:
        i = 0
        while i < len(words):
            for j in range(i+1, len(words)+1):
                phrase = ' '.join(words[i:j])
                if goldphrase.has_key(phrase):
                    #print('tmp:', phrase)
                    if j+1 >= len(words) or (j+1 < len(words) and not goldphrase.has_key(' '.join(words[i:(j+1)]))):
                        new_gold = [i, j, phrase, goldphrase[phrase]]
                        found_gold.append(new_gold)
                        for k in range(i, j):
                            golden[k] =1
                        for k in range(i, j-1):
                            if not words[k].endswith('@@'):
                                words_result[k] = words_result[k]+'@@'
                        i = j
                        result[transform_gold(phrase)] = [[goldphrase[phrase], 1.]]
                        break
            i += 1
        print('found gold:', found_gold)

    for i in range(len(words)):
        if golden[i] == 1:
            continue
        #if cov:
        #    if cov[i] != 0:
        #        continue
        for j in range(i+1, min(i+ngram+1, len(words)+1)):
            if golden[j-1] == 1:
                break
            #if cov:
            #    if cov[j-1] != 0:
            #        break
            phrase = ' '.join(words[i:j])
            if phrases.has_key(phrase):
                if params.allow_src2stop:
                    result[phrase] = phrases[phrase]
                else:
                    result[phrase] = filter_stop(phrases[phrase], stopword_list)
                for k in range(i, j):
                    covered[k] = 1
    # special treatment for words with no phrase
    for i in range(len(words)):
        #if cov:
        #    if cov[i] != 0:
        #        continue
        if golden[i] == 1:
            continue
        '''
        if result.has_key(words[i]):
            if not [words[i], 1.0] in result[words[i]]:
                result[words[i]].append([words[i], 1.0])
        else:
            result[words[i]] = [[words[i], 1.0]]
        '''
        if result.has_key(words[i]):
            if words[i] == ',':
                result[words[i]].append([',', 1.0])
            if not ['<oov>', 1.0] in result[words[i]]:
                result[words[i]].append(['<oov>', 1.0])
        else:
            result[words[i]] = [['<oov>', 1.0]]
            
    if rbpe:
        result_rbpe = {}
        words_rbpe = reverse_bpe(' '.join(words)).split(' ')
        for key in result.keys():
            if not key in words_rbpe:
                continue
            result_rbpe[reverse_bpe(key)] = result[key]
        for i in range(len(words_rbpe)):
            if result_rbpe.has_key(words_rbpe[i]):
                if not [words_rbpe[i], 1.0] in result_rbpe[words_rbpe[i]]:
                    result_rbpe[words_rbpe[i]].append([words_rbpe[i], 1.0])
            else:
                result_rbpe[words_rbpe[i]] = [[words_rbpe[i], 1.0]]
        return result_rbpe, golden, words_result
    else:
        return result, golden, words_result


def print_phrases(phrases):
    for k in phrases.keys():
        tmp = k+' ||| '
        for t in phrases[k]:
            if type(t) is list:
                tmp += t[0]+' '+str(t[1])+' ||| '
        print(tmp.encode('utf-8'))


cdef print_stack(translation_status *stack, int len_src, int count):
    cdef int i
    cdef int j
    for i in range(count):
        printf('Number %d:\n', i)
        printf('translation: %s\n', stack[i].translation)
        printf('coverage: [')
        for j in range(len_src):
            printf('%d ',stack[i].coverage[j])
        printf(']\n')
        printf('align_loss: %f\n', stack[i].align_loss)
        printf('src2null_loss: %f\n', stack[i].src2null_loss)
        printf('translation_loss: %f\n', stack[i].translation_loss)
        printf('previous: [%d %d %d]\n', stack[i].previous[0], stack[i].previous[1], stack[i].previous[2])
        printf('loss: %f\n\n', stack[i].loss)


cdef print_stack_finished(translation_status *stack, int count):
    cdef int i
    for i in range(count):
        printf('end%d: %s %f\n', i, stack[i].translation, stack[i].loss)


def find(words, cov, k):
    kwords = k.split(' ')
    pos = 0
    kpos = 0
    while pos < len(words):
        if words[pos] == kwords[kpos] and cov[pos] == 0:
            kpos += 1
            pos += 1
            if kpos == len(kwords):
                return [pos-len(kwords), pos]
        else:
            kpos = 0
            pos += 1
    return -1



def clean(stacks, i):
    if i < 0:
        return stacks
    for j in range(len(stacks[i])):
        for k in range(len(stacks[i][j])):
            stacks[i][j][k][2] = None
    return stacks


def generate_new(words_src, phrases, stack, length):
    result = []
    for tmp in stack:
        part, cov, loss = tmp
        for k in phrases.keys():
            found = find(words_src, cov, k)
            if found != -1:
                cov_new = copy.deepcopy(cov)
                for i in range(found[0], found[1]):
                    cov_new[i] = 1
                finish = is_finish(cov_new)
                for p in phrases[k]:
                    if not finish:
                        result.append([(part+' '+p).strip(), cov_new, length+len(k.split(' '))])
                    else:
                        result.append([(part+' '+p+' <eos>').strip(), cov_new, length+len(k.split(' '))])
    return result


cdef int getid_word(ivocab, word):
    cdef int result
    if not word in ivocab:
        result = ivocab['<unk>']
    else:
        result = ivocab[word]
    return result


def getid(ivocab, text):
    if text == '':
        return [ivocab['<eos>']]
    words = text.split(' ')
    result = []#[ivocab['<bos>']]
    for word in words:
        if not word in ivocab:
            result.append(ivocab['<unk>'])
        else:
            result.append(ivocab[word])
    result.append(ivocab['<eos>'])
    return result


def remove_duplicate(inp):
    result = []
    for i in inp:
        if not i in result:
            result.append(i)
    return result


def build_ivocab(vocab):
    result = {}
    for num, word in enumerate(vocab):
        result[word] = num
    return result


def is_finish(element):
    cov = element[0]
    status = element[1]
    if status['limit'] == 'limited':
        return False
    for i in cov:
        if i != 1:
            return False
    return True


def load_null2trg(n2tfile):
    content = open(n2tfile, 'r').read()
    result = content.split('\n')
    if result[-1] == '':
        del result[-1]
    return result


def load_goldphrase(goldfile):
    content = open(goldfile, 'r').read()
    pairs = content.split('\n')
    if pairs[-1] == '':
        del pairs[-1]
    result = {}
    for pair in pairs:
        src, trg = pair.split('\t')
        src = src.decode('utf-8')
        trg = trg.decode('utf-8')
        result[src] = trg
    return result
    

def getstate(encoderstate, num_layers):
    result = {"layer_%d" % i: encoderstate["decoder"]["layer_%d" % i]
              for i in range(num_layers)}
    return result


cdef merge_tensor(stack, layer):
    cdef int i
    result = {}
    result["key"] = np.concatenate([stack[i]["layer_%d" % layer]["key"] for i in range(len(stack))])
    result["value"] = np.concatenate([stack[i]["layer_%d" % layer]["value"] for i in range(len(stack))])
    return result


def outdims(state, num_layers):
    result = []
    shape = state["decoder"]["layer_0"]["key"].shape
    for i in range(shape[0]):
        tmp = {
            "layer_%d" % j :{
                "key": state["decoder"]["layer_%d" % j]["key"][i:i+1],
                "value": state["decoder"]["layer_%d" % j]["value"][i:i+1]
            }
            for j in range(num_layers)}
        result.append(tmp)
    return result


def append_empty(stacks, length, cov_num):
    while len(stacks) <= length:
        stacks.append([])
    while len(stacks[length]) <= cov_num:
        stacks[length].append([])
    return stacks


cdef int can_addstack(translation_status *stack, int stack_count, float loss, int beam_size, float align_loss=-1):
    cdef int i
    if stack_count < beam_size:
        return 1
    else:
        if loss > stack[beam_size-1].loss:
            return 1
        elif loss == stack[beam_size-1].loss:
            if align_loss > 0:
                for i in range(stack_count):
                    if loss == stack[i].loss:
                        # pending: consider multiple status
                        # warning: only allow keep_status_num == 1 here
                        if align_loss > stack[i].align_loss:
                            return 1
            else:
                return 1
    return 0


cdef get_kmax(sorted_array, int *avail, int num, automatons_state state, golden=None):
    result = []
    cdef int pos = 0
    cdef int isvisible = 0
    cdef int i
    while len(result) < num and pos < len(sorted_array):
        if avail[sorted_array[pos]] == 0:
            isvisible = 0
            for i in range(state.num_visible):
                if sorted_array[pos] == state.visible[i]:
                    isvisible = 1
                    break
            if isvisible:
                if golden[sorted_array[pos]] == 0:
                    result.append(sorted_array[pos])
        pos += 1
    return result


def sorted_index(inp):
    tosort = []
    for i in range(len(inp)):
        tosort.append([inp[i], i])
    tosort = sorted(tosort, key=lambda x: x[0], reverse=True)
    result = [i[1] for i in tosort]
    return result


def argmax(array, avail, num):
    tosort = []
    for i in range(len(array)):
        if avail[i] == 0:
            tosort.append([array[i], i])
    tosort = sorted(tosort, key=lambda x: x[0], reverse=True)
    result = [i[1] for i in tosort]
    return result[:num]
    

def reverse_bpe(inp):
    return inp.replace('@@ ', '')


def min_align_prob(statuses):
    which = -1
    minprob = 0
    for key in statuses.keys():
        if get_align_prob(key) < minprob:
            minprob = get_align_prob(key)
            which = key
    return minprob, which


def get_align_prob(status):
    tmp = json.loads(status)
    return tmp['align_prob']


cdef int is_equal(float a, float b):
    if abs(a-b) < 1e-6:
        return 1
    return 0


cdef int add_stack_limited(translation_status *stack_limit, int stack_limit_count, translation_status element, int len_src, params):
    stack_limit[stack_limit_count] = element
    stack_limit_count += 1
    return stack_limit_count 


cdef int add_stack(translation_status *stack, int stack_count, translation_status element, int len_src, int beam_size, merge_status=None, max_status=1, verbose=0):
    cdef int i, j 
    cdef translation_status tmp
    '''
    if strcmp(element.translation, 'after the peace summit between leaders of south and north korea in 2000 ,') == 0:
        printf('testing: %f; %f\n', element.loss, element.align_loss)
    if verbose == 1:
        printf('add stack: %s\n', element.translation)
        printf('stack count %d/%d\n', stack_count, beam_size)
        if stack_count > 0:
            printf('add stack first: %s\n', stack[0].translation)
    '''
    if merge_status:
        for i in range(stack_count):
            #if element.translation == stack[i].translation: 
            if strcmp(element.translation, stack[i].translation) == 0: 
                #printf("add stack found same %d\n", i)
                if max_status == 1:
                    #assert stack[i].loss == element.loss
                    if stack[i].loss < element.loss:
                        #for j in range(len_src):
                        #    stack[i].coverage[j] = element.coverage[j]
                        stack[i].coverage = element.coverage
                        stack[i].align_loss = element.align_loss
                        stack[i].automatons = element.automatons
                        stack[i].limited = element.limited
                        stack[i].limits = element.limits
                        stack[i].previous = element.previous
                        stack[i].loss = element.loss
                        j = i
                        while j > 0 and stack[j].loss > stack[j-1].loss:
                            tmp = stack[j-1]
                            stack[j-1] = stack[j]
                            stack[j] = tmp
                            j -= 1
                    elif is_equal(stack[i].loss, element.loss) == 1 and element.align_loss > stack[i].align_loss:
                        #for j in range(len_src):
                        #    stack[i].coverage[j] = element.coverage[j]
                        stack[i].coverage = element.coverage
                        stack[i].align_loss = element.align_loss
                        stack[i].automatons = element.automatons
                        stack[i].limited = element.limited
                        stack[i].limits = element.limits
                        stack[i].previous = element.previous
                return stack_count 
    cdef int pos = 0
    #printf("add stack not same\n")
    if stack_count < beam_size:
        #printf("add stack not full\n")
        while pos < stack_count and element.loss < stack[pos].loss:
            pos += 1
        if pos < stack_count:
            for i in range(stack_count, pos, -1):
                stack[i] = stack[i-1]
        #pos = stack_count
        stack[pos] = element
        stack_count += 1
        return stack_count 
    else:
        #printf("add stack full\n")
        while pos < beam_size and element.loss < stack[pos].loss:
            pos += 1
        if pos < beam_size:
            for i in range(beam_size-1, pos, -1):
                stack[i] = stack[i-1]
            #stack[pos] = copy_translation_status(element, len_src)
            stack[pos] = element
        #printf('add stack pos %d\n', pos)
        return stack_count 


def list_of_empty_list(num):
    result = []
    for i in range(num):
        result.append([])
    return result


cdef char* get_first_word_and_length(char *inp, int *have_first):
    cdef int pos = 0, found_space = 0
    cdef int length = strlen(inp)
    while pos < length:
        if inp[pos] == ' ':
            found_space = 1
            break
        pos += 1
    if found_space == 1:
        result = <char*> malloc((pos+1)*sizeof(char))
        strncpy(result, inp, pos)
        result[pos] = 0
        have_first[0] = pos
    else:
        result = <char*> malloc((length+1)*sizeof(char))
        strcpy(result, inp)
        result[length] = 0
        have_first[0] = -1
    return result


cdef char* get_first_word(char *inp):
    cdef int pos = 0, found_space = 0
    cdef int length = strlen(inp)
    cdef char *result
    while pos < length:
        if inp[pos] == ' ':
            found_space = 1
            break
        pos += 1
    if found_space == 1:
        result = <char*> malloc((pos+1)*sizeof(char))
        strncpy(result, inp, pos)
        result[pos] = 0
        return result
    else:
        result = <char*> malloc((length+1)*sizeof(char))
        strcpy(result, inp)
        result[length] = 0
        return result


cdef int compare_candidate(candidate a, candidate b):
    if a.loss > b.loss:
        return 1
    elif a.loss == b.loss:
        if strcmp(get_first_word(a.phrase), get_first_word(b.phrase)) == 0:
            if a.prob_align > b.prob_align:
                return 2
            elif a.prob_align == b.prob_align:
                return 100
            else:
                return -2
        if a.prob_align > b.prob_align:
            return 1
        elif a.prob_align == b.prob_align:
            return 0
        else:
            return -1
    else:
        return -1


cdef int add_candidate_limit(candidate *candidate_list_limit, int count, candidate new, params):
    candidate_list_limit[count] = new
    return count+1 


cdef int add_candidate(candidate *candidate_list, int count, candidate new, params):
    cdef int i, pos, comp
    cdef int have_same = 0
    '''
    for i in range(count):
        if strcmp(get_first_word(candidate_list[i].phrase), get_first_word(new.phrase)) == 0:
            if new.prob_align > candidate_list[i].prob_align:
                candidate_list[i] = new
                return count
            else:
                return count
    '''
    if count < params.beam_size:
        pos = count 
        while pos >= 1:
            comp = compare_candidate(candidate_list[pos-1], new)
            if comp == 1:
                break
            if comp == -2:
                candidate_list[pos-1] = new
                return count
            elif comp == 2 or comp == 100:
                return count
            pos -= 1
        if pos < count:
            for i in range(count, pos, -1):
                candidate_list[i] = candidate_list[i-1]
        candidate_list[pos] = new
        return count+1
    else:
        pos = params.beam_size
        while pos >=1:
            comp = compare_candidate(candidate_list[pos-1], new)
            if comp == 1:
                break
            if comp == -2:
                candidate_list[pos-1] = new
                return count
            elif comp == 2 or comp == 100:
                return count
            pos -= 1
        if pos < params.beam_size:
            for i in range(params.beam_size-1, pos, -1):
                candidate_list[i] = candidate_list[i-1]
            candidate_list[pos] = new
        return count


def get_translate_status(words_src, phrases, phrases_reverse, part, ngrams):
    '''
        output: [can_finish, word_options]
    '''
    trg_words = part.split(' ')
    options = []
    for k in phrases_reverse.keys():
        options.append(k)
    count = {}
    for word in words_src:
        if not word in count:
            count[word] = [0,1]
        else:
            count[word][1] += 1
    for ng in range(1, ngrams):
        for pos in range(len(trg_words)-ng):
            p = ' '.join(trg_words[pos:pos+ng])
            if phrases_reverse.has_key(p):
                for k in phrases_reverse[p]:
                    count[k][0] += 1
    can_finish = True
    for k in count.keys():
        if count[k][0] < count[k][1]:
            can_finish = False
            break
    return can_finish, options


def get_src2null_prob(src2null_prob, word):
    if src2null_prob.has_key(word):
        return src2null_prob[word][2]
    else:
        return 0.


cdef float get_lp(int length, float alpha):
    return pow((5.0 + length) / 6.0, alpha)


time_totalsp = 0
'''
cdef translation_status to_finish(translation_status state, int length, float alpha):
    result = []
    result.append(state[0])
    length = len(state[0].split(' '))
    length_penalty = get_lp(length, alpha)
    result.append(state[3])
    result.append(state[-1] / length_penalty)
    #result.append(state[-1] / length)
    return result
'''

def merge_duplicate(stack):
    rid = {}
    result = []
    for i in range(len(stack)):
        part = stack[i][0]
        if rid.has_key(part):
            rid[part].append(i)
        else:
            rid[part] = [i]
            
    for k in rid.keys():
        tmp = stack[rid[k][0]]
        tmp[1] = [tmp[1]]
        for i in range(1, len(rid[k])):
            n = stack[rid[k][i]]
            if not n[1] in tmp[1]:
                tmp[1] += [n[1]]
        result.append(tmp)
    return result

cdef float my_log(float x):
    if x == 0:
        return -10000
    else:
        return log(x)

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.checkpoint, args.model, params)
    override_parameters(params, args)

    ### C version params ###
    cdef params_s params_c
    params_c.beam_size = params.beam_size
    params_c.keep_status_num = params.keep_status_num
    params_c.split_limited = params.split_limited
    params_c.src2null_loss = params.src2null_loss
    params_c.bpe_phrase = params.bpe_phrase
    ### declaration for Cython ###
    cdef:
        # params
        int decode_length = params.decode_length
        float decode_alpha = params.decode_alpha
        # universal
        int max_len_trg = 100
        int max_len_src = 100
        int max_limit = 10000
        int max_candidate = 10000
        int i, j, k, len_tmp, pos, pos_end, offset
        char *tmpstring
        phrase_pair *phrases_c
        automatons newautom
        # encode & prepare
        float time_test[20]
        int count_test[20]
        int len_src, length
        int *coverage
        translation_status element_init
        translation_status stacks[100][100][4]
        int stacks_count[100][100]
        translation_status *stacks_limit[100][100]
        int stacks_limit_count[100][100]
        translation_status finished[4]
        int finished_count
        # src2null
        int num_cov
        int all_empty
        translation_status element
        translation_status newelement
        float new_loss, new_word_loss
        float new_align_loss
        # neural
        int maps[100][4]
        int maps_limit[100][10000]
        int num_x
        # generation
        int len_covered, nosense, is_visible, num_total, tmp_id, have_first, total_length, len_first
        float prob_align, total_src2null_loss
        beam add_result
        char *tmpstr2
        char *newbuffer
        char *firstword
        char *firstword2
        automatons_state autostate
        candidate candidate_phrase_list[100][4]
        candidate *candidate_phrase_list_limit[100]
        candidate new_candidate
        int candidate_phrase_list_count[100]
        int candidate_phrase_list_limit_count[100]
        float length_penalty
    ###

    # Build Graph
    with tf.Graph().as_default():
        model = model_cls(params)
        #print('input file:', args.input)
        inputs = read_files([args.input])[0]
        #print('inputs', inputs)

        # Load phrase table
        #if args.tmpphrase and os.path.exists(args.tmpphrase):
        #    print('load tmpphrase')
        #    phrase_table = json.load(open(args.tmpphrase, 'r'))
        #else:
        phrase_table = json.load(open(args.phrase, 'r'))
        src2null_prob = json.load(open(args.nullprob ,'r'))
        if args.null2trg:
            null2trg_vocab = load_null2trg(args.null2trg)
            print('null2trg vocab:', null2trg_vocab)
        stoplist = load_null2trg(args.stoplist)
        print('stoplist:', stoplist)
        goldphrase = None
        if params.use_golden:
            goldphrase = load_goldphrase(args.goldphrase)
            print('golden phrase:', goldphrase)
        punc = None
        if params.punc_border:
            punc = automatons_o.load_punc(args.puncfile)
            print('puncs:', punc)

        # Load ivocab
        ivocab_src = build_ivocab(params.vocabulary["source"])
        ivocab_trg = build_ivocab(params.vocabulary["target"])
        
        #print(features)
        score_fn = model.get_evaluation_func()
        score_cache_fn = model.get_evaluation_cache_func()
        placeholder = {}
        placeholder["source"] = tf.placeholder(tf.int32, [None,None], "source")
        placeholder["source_length"] = tf.placeholder(tf.int32, [None], "source_length")
        placeholder["target"] = tf.placeholder(tf.int32, [None,None], "target")
        placeholder["target_length"] = tf.placeholder(tf.int32, [None], "target_length")
        scores = score_fn(placeholder, params)
        state = {
            "encoder": tf.placeholder(tf.float32, [None, None, params.hidden_size], "encoder"),
            "decoder": {"layer_%d" % i: {
                "key": tf.placeholder(tf.float32, [None, None, params.hidden_size], "decoder_key"),
                "value": tf.placeholder(tf.float32, [None, None, params.hidden_size], "decoder_value")
            } for i in range(params.num_decoder_layers) }
        }
        scores_cache = score_cache_fn(placeholder, state, params)

        # create cache
        enc_fn, dec_fn = model.get_inference_func()
        p_enc = {}
        p_enc["source"] = tf.placeholder(tf.int32, [None,None], "source")
        p_enc["source_length"] = tf.placeholder(tf.int32, [None], "source_length")
        enc = enc_fn(placeholder, params)
        dec = dec_fn(placeholder, state, params)

        sess_creator = tf.train.ChiefSessionCreator(
            config=session_config(params)
        )

        # Load checkpoint
        tf.logging.info("Loading %s" % args.checkpoint)
        var_list = tf.train.list_variables(args.checkpoint)
        values = {}
        reader = tf.train.load_checkpoint(args.checkpoint)

        for (name, shape) in var_list:
            if not name.startswith(model_cls.get_name()):
                continue

            tensor = reader.get_tensor(name)
            values[name] = tensor

        ops = set_variables(tf.trainable_variables(), values,
                            model_cls.get_name())
        #scores = score_fn(features, params)

        sess_creator = tf.train.ChiefSessionCreator(
            config=session_config(params)
        )

        # Load checkpoint
        tf.logging.info("Loading %s" % args.checkpoint)
        var_list = tf.train.list_variables(args.checkpoint)
        values = {}
        reader = tf.train.load_checkpoint(args.checkpoint)

        for (name, shape) in var_list:
            if not name.startswith(model_cls.get_name()):
                continue

            tensor = reader.get_tensor(name)
            values[name] = tensor

        ops = set_variables(tf.trainable_variables(), values,
                            model_cls.get_name())
        assign_op = tf.group(*ops)

        # Create session
        sess = tf.train.MonitoredSession(session_creator=sess_creator)
        sess.run(assign_op)

        fd = tf.gfile.Open(args.output, "w")

        fout = open(args.output, 'w')
        count = 0


        for input in inputs:
            time_test = [0]*20
            time_test_tag = [0]*20
            count_test = [0]*20
            count_test_tag = [0]*20
            ### testing tag
            time_test_tag = ['src2null_inner', 'src2null_inner_afterprune', 'neural_prepare', 'neural_decoding', 'neural_after', 'limit_inner', 'generate candidate', 'generate from phrase', 'generate from word', 'generate stop word', 'process candidate', 'candidate_normal', 'candidate_inter', 'candidate_limit', 'candidate_limit_afterprune', 'time_eos',0,'time_testing','before_encode','after encoder before decode']
            count_test_tag = ['src2null_inner', 'src2null_inner_afterprune', 'neural_input', 'limit_inner', 'phrase_inner', 'word_inner', 'stopword', 'candidate_normal', 'candidate_normal_afterprune', 'candidate_limit', 'candidate_limit_afterprune', 0,0,0,0,0,0,'count_testing',0,0]
            ###
            count += 1
            start = time.time()
            src = copy.deepcopy(input)
            src = src.decode('utf-8')
            words = src.split(' ')
            len_src = len(words)
            probs_null = [get_src2null_prob(src2null_prob, w) for w in words]
            null_order = sorted_index(probs_null)
            f_src = {}
            f_src["source"] = [getid(ivocab_src, input) ]
            f_src["source_length"] = [len(f_src["source"][0])] 
            #print('input_enc', f_src)
            feed_src = {
                placeholder["source"]: f_src["source"],
                placeholder["source_length"] : f_src["source_length"]
            }
            time_enc_start = time.time()
            time_test[18] = time_enc_start-start
            encoder_state = sess.run(enc, feed_dict=feed_src)
            time_enc_end = time.time()
            time_encode = time_enc_end-time_enc_start

            # generate a subset of phrase table for current translation
            phrases, golden, words_result = subset(phrase_table, words, args.ngram, params, rbpe=args.rbpe, stopword_list=stoplist, goldphrase=goldphrase)
            phrases_c = build_phrases_c(phrases)
            words = words_result
            phrases_reverse = reverse_phrase(phrases)
            #print('reverse phrase:', phrases_reverse)
            print(count)
            print('source:', src.encode('utf-8'))
            if args.verbose:
                print('golden:', golden)
                print('probs_null:', probs_null)
            if args.rbpe:
                words = reverse_bpe(src).split(' ')
                print('reverse_bpe:', reverse_bpe(src).encode('utf-8'))
            coverage = <int *> malloc(len_src * sizeof(int))
            for i in range(len_src):
                coverage[i] = 0
            
            #if args.tmpphrase:
            #    json.dump(phrases, open(args.tmpphrase, 'w'))
            if args.verbose:
                print_phrases(phrases)

            state_init = {}
            state_init["encoder"] = encoder_state 
            for i in range(params.num_decoder_layers):
                state_init["decoder"] = {}
                state_init["decoder"]["layer_%d" % i] = np.zeros((0, params.hidden_size))
            '''
            stacks:
            1. partial translation
            2. coverage status (dict)
            3. hidden state ({"layer_0": [...], "layer_1": [...]})
            4. [last_trg_len , last_cov_num, last beam id, aggregated_nullprob]
            5. score
            '''
            #init_status = [coverage, ['normal', ''], 0.]
            autom = automatons_build(words, punc, params)
            print_autom(autom)
            #autom = automatons_o.build(words, params)
            #automatons_o.print_autom(autom)
            #init_status = {'coverage': coverage, 'limit': ['normal', ''], 'align_prob': 0., 'automatons': 0}
            #if params.keep_status_num == 1:
            #    element_init = ['', init_status, getstate(encoder_state, params.num_decoder_layers), [0, 0, 0, 0.], 0]
            #else:
            #    element_init = ['', {json.dumps(init_status): 1}, getstate(encoder_state, params.num_decoder_layers), [0, 0, 0, 0.], 0]
            element_init.translation = ''
            element_init.coverage = coverage
            element_init.align_loss = 0
            element_init.src2null_loss = 0
            element_init.translation_loss = 0
            element_init.automatons = 0
            element_init.limited = 0
            element_init.limits = ''
            element_init.hidden_state_id = 0
            hidden_state_pool = [getstate(encoder_state, params.num_decoder_layers)]
            element_init.previous = [0, 0, 0]
            element_init.loss = 0
            stacks[0][0][0] = element_init
            stacks_count = [[0]*100]*100
            stacks_count[0][0] = 1
            stacks_limit_count = [[0]*100]*100
            for i in range(100):
                for j in range(100):
                    free(stacks_limit[i][j])
                    stacks_limit[i][j] = <translation_status*> malloc(max_limit * sizeof(translation_status))
            finished_count = 0
            length = 0

            time_neural = 0
            time_null = 0
            time_generate = 0
            time_limit = 0
            time_prepare_end = time.time()
            time_prepare = time_prepare_end-start
            time_test[19] = time_prepare_end-time_enc_end
            while True:
                # source to null
                time_null_start = time.time()
                if args.verbose:
                    printf('=== length: %d ===\n', length)
                    printf('== src2null ==\n')
                all_empty = 1
                for num_cov in range(0, len_src+1):
                    if stacks_count[length][num_cov] == 0:
                        continue
                    if args.verbose:
                        printf('=%d/%d=\n',length,num_cov)
                    all_empty = 0

                    if args.verbose:
                        printf('normal\n')
                        print_stack(stacks[length][num_cov], len_src, stacks_count[length][num_cov])
                        printf('limited: %d\n', stacks_limit_count[length][num_cov])
                        #print_stack(stacks_limit[length][num_cov])
                    for i in range(stacks_count[length][num_cov]):
                        element = stacks[length][num_cov][i]
                        autostate = autom.states[element.automatons]
                        if can_addstack(stacks[length][num_cov+1], stacks_count[length][num_cov+1], element.loss, params_c.beam_size) == 0:
                            continue
                        if element.limited == 1:
                            continue
                        time_0s = time.time()
                        # not optimized
                        kmax = get_kmax(null_order, element.coverage, params_c.beam_size, autostate, golden=golden)
                        for nullpos in kmax:
                            count_test[0] += 1
                            
                            assert element.coverage[nullpos] == 0
                            if params_c.src2null_loss:
                                total_src2null_loss = element.src2null_loss+my_log(probs_null[nullpos])
                                if total_src2null_loss < -1*length:
                                    continue
                                new_loss = element.loss+my_log(probs_null[nullpos])
                            else:
                                new_loss = element.loss
                                #total_src2null_loss = 0
                            new_align_loss = element.align_loss+my_log(probs_null[nullpos])
                            if can_addstack(stacks[length][num_cov+1], stacks_count[length][num_cov+1], new_loss, params_c.beam_size, align_loss=new_align_loss) == 0:
                                continue
                            count_test[1] += 1
                            time_1s = time.time()
                            newelement = copy_translation_status(element, len_src)
                            newelement.coverage[nullpos] = 1
                            newelement.align_loss = new_align_loss
                            if can_go_next(autostate, newelement.coverage) == 1 and autostate.next_state != -1:
                                newelement.automatons = autostate.next_state

                            newelement.previous = [length, num_cov, i]
                            newelement.loss = new_loss
                            newelement.src2null_loss = total_src2null_loss
                            #printf('%d/%d add to %d/%d\n', length, num_cov, length, num_cov+1)
                            stacks_count[length][num_cov+1] = add_stack(stacks[length][num_cov+1], stacks_count[length][num_cov+1], newelement, len_src, params_c.beam_size, params.merge_status, params_c.keep_status_num)
                            time_1e = time.time()
                            time_test[1] += time_1e-time_1s
                            break
                        time_0e = time.time()
                        time_test[0] += time_0e-time_0s
                time_null_end = time.time()
                time_null += time_null_end-time_null_start

                if all_empty == 1:
                    break

                # do the neural
                #printf('start neural prepare\n')
                time_neural_start = time.time()
                neural_result = [0]*(len_src+1)
                neural_result_limit = [0]*(len_src+1)
                time_ts = time.time()
                features, num_x = get_feature_map(stacks[length], stacks_count[length], stacks_limit[length], stacks_limit_count[length], len_src, ivocab_trg, hidden_state_pool, params, maps, maps_limit)
                time_te = time.time()
                time_test[17] += time_te-time_ts
                count_test[2] += num_x
                features["encoder"] = np.tile(encoder_state["encoder"], (num_x, 1, 1))
                features["source"] = [getid(ivocab_src, input)] * num_x 
                features["source_length"] = [len(features["source"][0])] * num_x 
                feed_dict = {
                    placeholder['source']: features['source'],
                    placeholder['source_length']: features['source_length'],
                    placeholder['target']: features['target'],
                    placeholder['target_length']: features['target_length']}
                dict_tmp={state["encoder"]: features["encoder"]}
                feed_dict.update(dict_tmp)
                dict_tmp={state["decoder"]["layer_%d" % i]["key"]: features["decoder"]["layer_%d" % i]["key"] for i in range(params.num_decoder_layers)}
                feed_dict.update(dict_tmp)
                dict_tmp={state["decoder"]["layer_%d" % i]["value"]: features["decoder"]["layer_%d" % i]["value"] for i in range(params.num_decoder_layers)}
                feed_dict.update(dict_tmp)
                time_2e = time.time()
                time_test[2] += time_2e-time_neural_start

                #printf('start neural\n')
                log_probs, new_state = sess.run(dec, feed_dict=feed_dict)
                time_3e = time.time()
                time_test[3] += time_3e-time_2e
                new_state = outdims(new_state, params.num_decoder_layers)
                hidden_state_pool = new_state
                for num_cov in range(0, len_src+1):
                    if stacks_count[length][num_cov] == 0:
                        continue
                    neural_result[num_cov] = [[], []]
                    for i in range(stacks_count[length][num_cov]):
                        pos = maps[num_cov][i]
                        neural_result[num_cov][0].append(log_probs[pos])
                        neural_result[num_cov][1].append(pos)
                for num_cov in range(0, len_src+1):
                    if stacks_limit_count[length][num_cov] == 0:
                        continue
                    neural_result_limit[num_cov] = [[], []]
                    for i in range(stacks_limit_count[length][num_cov]):
                        pos = maps_limit[num_cov][i]
                        neural_result_limit[num_cov][0].append(log_probs[pos])
                        neural_result_limit[num_cov][1].append(pos)
                #print('neural_result', neural_result)
                #print('neural_result_limit', neural_result_limit)
                time_neural_end = time.time()
                time_neural += time_neural_end-time_neural_start
                time_test[4] += time_neural_end-time_3e

                # update the stacks
                if args.verbose:
                    printf('== generation ==\n')
                    printf('generate from limited\n')
                time_generate_start = time.time()
                # limited stack
                for num_cov in range(0, len_src+1):
                    #printf('= limit %d =\n', num_cov)
                    if neural_result_limit[num_cov] == 0:
                        continue
                    log_probs_limit, new_state_limit_id = neural_result_limit[num_cov]
                    if params_c.split_limited:
                        for i in range(stacks_limit_count[length][num_cov]):
                            element = stacks_limit[length][num_cov][i]
                            #assert sum(element[1]['coverage']) == num_cov
                            time_5s = time.time()
                            for nosense in range(1):
                                count_test[3] += 1
                                #assert status['limit'][0] == "limited"
                                #printf('point 1: %s\n', element.limits)
                                pos = -1
                                still_limited = 0
                                for j in range(strlen(element.limits)):
                                    if element.limits[j] == ' ':
                                        pos = j
                                        still_limited = 1
                                        break
                                #printf('point 1.1: %d %d\n', strlen(element.limits), pos)

                                    
                                if still_limited:
                                    tmpstr2 = <char*> malloc(pos+1)
                                    strncpy(tmpstr2, element.limits, pos)
                                    tmpstr2[pos] = 0
                                else:
                                    tmpstr2 = element.limits
                                new_word_loss = log_probs_limit[i][getid_word(ivocab_trg, tmpstr2)]
                                new_loss = new_word_loss+element.loss

                                if still_limited == 1:
                                    #printf('still_limited\n')
                                    newelement = copy_translation_status(element, len_src)
                                    newbuffer = <char*> malloc(pos+2+strlen(newelement.translation))
                                    if length > 0:#strcmp(newelement.translation, '') != 0:
                                        strcpy(newbuffer, newelement.translation)
                                        strcat(newbuffer, ' ')
                                        strcat(newbuffer, tmpstr2)
                                        newbuffer[strlen(newelement.translation)+pos+1] = 0
                                    else:
                                        strncpy(newbuffer, element.limits, pos)
                                        newbuffer[pos] = 0
                                    newelement.translation = newbuffer 
                                    newbuffer = <char*> malloc(strlen(element.limits)+1)
                                    offset = pos+1
                                    newbuffer[strlen(element.limits)-offset] = 0
                                    strcpy(newbuffer, element.limits+offset)
                                    newelement.limits = newbuffer 
                                    newelement.hidden_state_id = new_state_limit_id[i]
                                    newelement.translation_loss += new_word_loss
                                    newelement.loss = new_loss
                                    # warning: only works when keep_status_num == 1
                                    stacks_limit_count[length+1][num_cov] = add_stack_limited(stacks_limit[length+1][num_cov], stacks_limit_count[length+1][num_cov], newelement, len_src, params)
                                else:
                                    #printf('not still_limited\n')
                                    if can_addstack(stacks[length+1][num_cov], stacks_count[length+1][num_cov], new_loss, params_c.beam_size) == 0:
                                        continue
                                    newelement = copy_translation_status(element, len_src)
                                    newbuffer = <char*> malloc(strlen(tmpstr2)+2+strlen(newelement.translation))
                                    #printf('point 3: %d ; %d ; %s ; %s\n', pos, strlen(tmpstr2), tmpstr2, newelement.limits)
                                    if length > 0:#strcmp(newelement.translation, '') != 0:
                                        strcpy(newbuffer, newelement.translation)
                                        strcat(newbuffer, ' ')
                                        strcat(newbuffer, tmpstr2)
                                        newbuffer[strlen(newelement.translation)+strlen(tmpstr2)+1] = 0
                                    else:
                                        strncpy(newbuffer, element.limits, strlen(tmpstr2))
                                        newbuffer[strlen(tmpstr2)] = 0
                                    newelement.translation = newbuffer 
                                    newelement.limited = 0
                                    newelement.limits = ""
                                    newelement.hidden_state_id = new_state_limit_id[i]
                                    newelement.translation_loss += new_word_loss
                                    newelement.loss = new_loss
                                    #printf('%d/%d add to %d/%d\n', length, num_cov, length+1, num_cov)
                                    stacks_count[length+1][num_cov] = add_stack(stacks[length+1][num_cov], stacks_count[length+1][num_cov], newelement, len_src, params_c.beam_size, params.merge_status, params_c.keep_status_num)
                            time_5e = time.time()
                            time_test[5] += time_5e-time_5s
                time_limit_end = time.time()
                time_limit += time_limit_end-time_generate_start

                #generate new options (unlimited)
                #printf('generating new options (unlimited)\n')
                for num_cov in range(0, len_src+1):
                    #print('= normal', num_cov, '=')
                    if neural_result[num_cov] == 0:
                        continue
                    log_probs, new_state_id = neural_result[num_cov]

                    for i in range(stacks_count[length][num_cov]):
                        element = stacks[length][num_cov][i]
                        for nosense in range(1):
                            visible = autom.states[element.automatons].visible
                            autostate = autom.states[element.automatons]
                            # limit (should not enter)
                            if element.limited == 1: 
                                assert 0 == 1
                            # no limitation
                            else:
                                # candidate phrase list: list of [phrase, pos_start, pos_end, loss, align_loss]
                                time_cs = time.time()
                                for j in range(len_src+1):
                                    free(candidate_phrase_list_limit[j])
                                    #candidate_phrase_list[j] = <candidate*> malloc(max_candidate*sizeof(candidate))
                                    candidate_phrase_list_limit[j] = <candidate*> malloc(max_candidate*sizeof(candidate))
                                    candidate_phrase_list_count[j] = 0
                                    candidate_phrase_list_limit_count[j] = 0
                                #candidate_phrase_list = list_of_empty_list(len_src+1)
                                #candidate_phrase_list_limit = list_of_empty_list(len_src+1)
                                all_covered = True
                                for j in range(autom.states[element.automatons].num_visible):
                                    pos = visible[j]
                                    # bpe_phrase
                                    time_phrase_start = time.time()
                                    if params_c.bpe_phrase:
                                        # find untranslated bpe phrase
                                        pos_end = pos
                                        if element.coverage[pos_end] == 1:
                                            continue
                                        is_visible = 0
                                        for k in range(autom.states[element.automatons].num_visible):
                                            if pos_end+1 == autom.states[element.automatons].visible[k]:
                                                is_visible = 1 
                                                break
                                        while is_visible == 1 and element.coverage[pos_end+1] == 0 and pos_end-pos < 3: #and words[pos_end].endswith('@@'):
                                            pos_end += 1
                                            if pos_end > pos:
                                                bpe_phrase = ' '.join(words[pos:pos_end+1])
                                                len_bpe_phrase = pos_end-pos+1
                                                if phrases.has_key(bpe_phrase):
                                                    # start translation
                                                    for j in range(len(phrases[bpe_phrase])):
                                                        # warning: need to build seperate candidate list for different number of covered words 
                                                        count_test[4] += 1
                                                        phrase, prob_align = phrases[bpe_phrase][j]
                                                        tmpstr2 = phrase
                                                        firstword = get_first_word_and_length(tmpstr2, &have_first)
                                                        tmp_id = getid_word(ivocab_trg, firstword)
                                                        new_loss = log_probs[i][tmp_id]
                                                        new_candidate.phrase = tmpstr2 
                                                        new_candidate.pos = pos
                                                        new_candidate.pos_end = pos_end
                                                        new_candidate.loss = new_loss
                                                        new_candidate.prob_align = prob_align
                                                        if params_c.split_limited and have_first != -1:
                                                            candidate_phrase_list_limit_count[len_bpe_phrase]= add_candidate_limit(candidate_phrase_list_limit[len_bpe_phrase], candidate_phrase_list_limit_count[len_bpe_phrase], new_candidate, params)
                                                            #printf('limit count: %d\n', candidate_phrase_list_limit_count[len_bpe_phrase])
                                                        else:
                                                            candidate_phrase_list_count[len_bpe_phrase] = add_candidate(candidate_phrase_list[len_bpe_phrase], candidate_phrase_list_count[len_bpe_phrase], new_candidate, params)
                                                        #printf('ADDED\n')
                                    time_phrase_end = time.time()
                                    time_test[7] += time_phrase_end-time_phrase_start
                                    #generate from source word
                                    if element.coverage[pos] == 0:
                                        all_covered = False
                                        if not phrases.has_key(words[pos]):
                                            continue
                                        num_total = len(phrases[words[pos]])
                                        for j in range(num_total):
                                            count_test[5] += 1
                                            phrase, prob_align = phrases[words[pos]][j] 
                                            tmpstr2 = phrase
                                            firstword = get_first_word_and_length(tmpstr2, &have_first) #0.25s/181w

                                            tmp_id = getid_word(ivocab_trg, firstword) # 0.35/181w
                                            new_loss = log_probs[i][tmp_id] 
                                            new_candidate.phrase = tmpstr2 
                                            new_candidate.pos = pos
                                            new_candidate.pos_end = pos
                                            new_candidate.loss = new_loss
                                            new_candidate.prob_align = prob_align
                                            if params_c.split_limited and have_first != -1:
                                                candidate_phrase_list_limit_count[1] = add_candidate_limit(candidate_phrase_list_limit[1], candidate_phrase_list_limit_count[1], new_candidate, params)
                                            else:
                                                candidate_phrase_list_count[1] = add_candidate(candidate_phrase_list[1], candidate_phrase_list_count[1], new_candidate, params)
                                    time_word_end = time.time()
                                    time_test[8] += time_word_end-time_phrase_end

                                # generate from null2trg
                                time_stop_start = time.time()
                                for j in range(len(null2trg_vocab)):
                                    stopword = null2trg_vocab[j]
                                    count_test[6] += 1
                                    tmpstr2 = stopword
                                    tmp_id = getid_word(ivocab_trg, tmpstr2)
                                    new_loss = log_probs[i][tmp_id]
                                    new_candidate.phrase = tmpstr2 
                                    new_candidate.pos = -1
                                    new_candidate.pos_end = -1
                                    new_candidate.loss = new_loss
                                    new_candidate.prob_align = 1
                                    candidate_phrase_list_count[0] = add_candidate(candidate_phrase_list[0], candidate_phrase_list_count[0], new_candidate, params)
                                time_ce = time.time()
                                time_test[6] += time_ce-time_cs
                                time_test[9] += time_ce-time_stop_start
                                    

                                #if args.verbose:
                                #    print('current:', element.translation.encode('utf-8'), element.loss)
                                #    print('candidates:', candidate_phrase_list)

                                #printf("processing candidates")
                                time_candidate_start = time.time()
                                for j in range(len_src):
                                    #printf("processing candidates %d/%d/%d\n", num_cov, i, j)
                                    #printf("normal\n")
                                    if num_cov+j > len_src:
                                        continue
                                    time_10s = time.time()
                                    for k in range(candidate_phrase_list_count[j]):
                                        cand = candidate_phrase_list[j][k]
                                        count_test[7] += 1
                                        pos = cand.pos
                                        pos_end = cand.pos_end
                                        loss = cand.loss
                                        prob_align = cand.prob_align
                                        firstword = get_first_word_and_length(cand.phrase, &have_first)
                                        new_loss = element.loss+loss
                                        # safe prune for search
                                        if finished_count >= params_c.beam_size and new_loss/get_lp(len_src+decode_length, decode_alpha) < finished[finished_count-1].loss:
                                            continue
                                        if can_addstack(stacks[length+1][num_cov+1], stacks_count[length+1][num_cov+1], new_loss, params_c.beam_size) == 0:
                                            break
                                        count_test[8] += 1
                                        newelement = copy_translation_status(element, len_src)
                                        if pos >= 0:
                                            len_covered = pos_end-pos+1
                                            for p in range(pos, pos_end+1):
                                                newelement.coverage[p] = 1
                                            if can_go_next(autostate, newelement.coverage) and autostate.next_state != -1:
                                                newelement.automatons = autostate.next_state
                                        else:
                                            len_covered = 0
                                        #assert len_covered == j
                                        #assert have_first == -1
                                        newelement.limited = 0
                                        newelement.limits = ' '

                                        len_first = strlen(firstword)
                                        newbuffer = <char*> malloc(len_first+2+strlen(newelement.translation))
                                        strcpy(newbuffer, newelement.translation)
                                        total_length = strlen(newelement.translation)
                                        if length > 0:#strcmp(newelement.translation, '') != 0:
                                            newbuffer[strlen(newelement.translation)] = ' '
                                            total_length += 1
                                        strcpy(newbuffer+total_length, firstword)
                                        total_length += len_first
                                        newbuffer[total_length] = 0
                                        newelement.translation = newbuffer 
                                        newelement.align_loss += log(prob_align)
                                        newelement.previous[0] = length
                                        newelement.previous[1] = num_cov
                                        newelement.previous[2] = i
                                        newelement.translation_loss += loss
                                        newelement.loss = new_loss
                                        newelement.hidden_state_id = new_state_id[i]
                                        stacks_count[length+1][num_cov+len_covered] = add_stack(stacks[length+1][num_cov+len_covered], stacks_count[length+1][num_cov+len_covered], newelement, len_src, params_c.beam_size, params.merge_status, params_c.keep_status_num)

                                    time_11e = time.time()
                                    time_test[11] += time_11e-time_10s
                                    if stacks_count[length+1][num_cov+j] > 0:
                                        loss_threshold = stacks[length+1][num_cov+j][stacks_count[length+1][num_cov+j]-1].loss
                                    else:
                                        loss_threshold = 0
                                    # limited
                                    time_13s = time.time()
                                    time_test[12] += time_13s-time_11e
                                    for k in range(candidate_phrase_list_limit_count[j]):
                                        cand = candidate_phrase_list_limit[j][k]
                                        count_test[9] += 1
                                        pos = cand.pos
                                        pos_end = cand.pos_end
                                        loss = cand.loss
                                        prob_align = cand.prob_align
                                        firstword = get_first_word_and_length(cand.phrase, &have_first)
                                        new_loss = element.loss+loss
                                        # safe prune for search
                                        if finished_count >= params_c.beam_size and new_loss/get_lp(len_src+decode_length, decode_alpha) < finished[finished_count-1].loss:
                                            continue
                                        # not safe prune
                                        if new_loss < loss_threshold:
                                            continue
                                        count_test[10] += 1
                                        #time_14s = time.time()
                                        newelement = copy_translation_status(element, len_src)
                                        if pos >= 0:
                                            len_covered = pos_end-pos+1
                                            for p in range(pos, pos_end+1):
                                                newelement.coverage[p] = 1
                                            if can_go_next(autostate, newelement.coverage) and autostate.next_state != -1:
                                                newelement.automatons = autostate.next_state
                                        else:
                                            len_covered = 0
                                        #assert len_covered == j
                                        newelement.limited = 1
                                        tmpstr2 = cand.phrase+have_first+1
                                        newbuffer = <char*> malloc(strlen(tmpstr2)+1)
                                        strcpy(newbuffer, tmpstr2)
                                        newbuffer[strlen(tmpstr2)] = 0
                                        newelement.limits = newbuffer
                                        newelement.align_loss += log(prob_align)
                                        newelement.translation_loss += loss
                                        newelement.loss = new_loss
                                        newelement.previous[0] = length
                                        newelement.previous[1] = num_cov
                                        newelement.previous[2] = i

                                        newbuffer = <char*> malloc(have_first+4+strlen(newelement.translation))
                                        strcpy(newbuffer, newelement.translation)
                                        total_length = strlen(newelement.translation)
                                        if length > 0:#strcmp(newelement.translation, '') != 0:
                                            newbuffer[strlen(newelement.translation)] = ' '
                                            total_length += 1
                                        strcpy(newbuffer+total_length, firstword)
                                        total_length += have_first
                                        newbuffer[total_length] = 0
                                        newelement.translation = newbuffer 
                                        newelement.hidden_state_id = new_state_id[i]
                                        stacks_limit_count[length+1][num_cov+len_covered] = add_stack_limited(stacks_limit[length+1][num_cov+len_covered], stacks_limit_count[length+1][num_cov+len_covered], newelement, len_src, params)
                                    time_13e = time.time()
                                    time_test[13] += time_13e-time_13s

                                time_candidate_end = time.time()
                                time_test[10] += time_candidate_end-time_candidate_start


                                if all_covered:
                                    new_loss = log_probs[i][getid_word(ivocab_trg, '<eos>')]
                                    new_loss += element.loss
                                    newelement = copy_translation_status(element, len_src)
                                    newbuffer = <char*> malloc(7+strlen(newelement.translation))
                                    strcpy(newbuffer, newelement.translation)
                                    strcat(newbuffer, ' <eos>')
                                    newbuffer[6+strlen(newelement.translation)] = 0
                                    newelement.translation = newbuffer 
                                    newelement.loss = new_loss
                                    length_penalty = get_lp(length+1, decode_alpha)
                                    newelement.loss /= length_penalty
                                    #print('to_finish:', new[0].encode('utf-8'), new[-1])
                                    finished_count = add_stack(finished, finished_count, newelement, len_src, params_c.beam_size)
                                time_eos_end = time.time()
                                time_test[15] += time_eos_end-time_candidate_end

                time_generate_end = time.time()
                time_generate += time_generate_end-time_generate_start
                if args.verbose:
                    #print('=',len_src, '=')
                    #print_stack(stacks[length][len_src])
                    print_stack_finished(finished, finished_count)
                length += 1
                if length > len_src+decode_length:
                    break
                #test
                #if length == 3:
                #    break

            if finished_count > 0:
                result = finished[0].translation
            else: 
                result = ''

            fout.write((result.replace(' <eos>', '').strip()+'\n').encode('utf-8'))
            print((result.replace(' <eos>', '').strip()).encode('utf-8'))

            end = time.time()
            global time_totalsp
            if args.time:
                print('time encode & prepare', time_prepare, 's')
                print('time encode neural', time_encode, 's')
                print('time total null:', time_null, 's')
                print('time total neural:', time_neural, 's')
                print('time total generate:', time_generate, 's')
                print('time total limit:', time_limit, 's')
                print('time total:', end-start, 'seconds')
                print('count:', zip(count_test_tag, count_test))
                print('time:', zip(time_test_tag, time_test))

            # rewinding
            '''
            if args.verbose:
                print('start rewinding...')
                print('src:', src.encode('utf-8'))
                print('trg:', finished[0][0].replace(' <eos>', '').strip().encode('utf-8'))
                lastpos = finished[0][1]
                print('first lastpos:', lastpos)
                words_trg = finished[0][0].replace(' <eos>', '').strip().split(' ')
                now = stacks[lastpos[0]][lastpos[1]][lastpos[2]]
                while True:
                    last_cov = now[1]['coverage']
                    last_words = len(now[0].split(' '))
                    lastpos = now[3]
                    print('lastpos:', lastpos)
                    now_words = lastpos[0]
                    #print(now_words, last_words)
                    trg_word = ' '.join(words_trg[now_words:last_words])
                    #print(lastpos)
                    now = stacks[lastpos[0]][lastpos[1]][lastpos[2]]
                    src_word = ''
                    for i in range(len(last_cov)):
                        if last_cov[i] == 1 and now[1]['coverage'][i] == 0:
                            src_word += ' '+words[i]
                    if src_word == '':
                        src_word = '<null>'
                    if trg_word == '':
                        trg_word = '<null>'
                    print(src_word.encode('utf-8'), '-', trg_word)
                    if lastpos[0] == 0 and lastpos[1] == 0:
                        break
            '''
                    
            '''
            if args.verbose:
                len_now = len(finished[0][0].split(' '))-1
                now = stacks[len_now][finished[0][1]]
                nowst = None
                while len_now > 0:
                    print("===", len_now, now[-1], "===")
                    print(now[0].encode('utf-8'), now[1], now[3])
                    find_translate(now, stacks[len_now-1][now[3][0]], nowst, now[3][1], words)
                    nowst = now[3][1]
                    now = stacks[len_now-1][now[3][0]]
                    len_now -= 1
            '''

if __name__ == "__main__":
    main(parse_args())
