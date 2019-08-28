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
import traceback
import re

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


cdef struct translation_status:
    char *translation
    int limited[2]
    int hidden_state_id
    int satisfied[20]
    float loss


cdef translation_status copy_translation_status(translation_status old, int len_src):
    cdef translation_status new
    new = old
    return new


cdef struct beam:
    translation_status *content
    int count


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
    parser.add_argument("--ngram", type=int, default=4,
                        help="ngram length")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    parser.add_argument("--tmpphrase", type=str, default="",
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
        decode_length=20
        # dba specific
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


def set_variables_new(var_list, value_dict, prefix, feed_dict):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                placeholder = tf.placeholder(tf.float32,
                                             name="placeholder/" + var_name)
                with tf.device("/cpu:0"):
                    op = tf.assign(var, placeholder)
                    ops.append(op)
                feed_dict[placeholder] = value_dict[name]
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


cdef get_feature_map(translation_status stack[150][4], int *stack_count, int num_constraints, ivocab_trg, hidden_state_pool, params, int maps[150][4], float threshold):
    #printf('get_feature_map\n')
    features = {}
    sentence_list = {}
    cdef int sentence_num = 0
    new_hidden_pool = []
    cdef int num_cov, pos, idx, i, max_len
    cdef int count = 0
    cdef translation_status element
    for num_cov in range(num_constraints+1):
        for idx in range(stack_count[num_cov]):
            count += 1
            element = stack[num_cov][idx]
            #if element.translation_loss < threshold:
            #    maps[num_cov][idx] = -1
            #    continue
            found = 0
            print('tmp:', num_cov, idx)
            print(element.limited[0])
            '''
            if sentence_list.has_key(element.translation):
                continue
                #maps[num_cov][idx] = sentence_list[element.translation]
            else:
                sentence_list[element.translation] = sentence_num
                maps[num_cov][idx] = sentence_num
                sentence_num += 1
                new_hidden_pool.append(hidden_state_pool[element.hidden_state_id])
            '''
    '''
    sen_ids_list = [0] * sentence_num
    for sen in sentence_list.keys():
        #assert sen_ids_list[sentence_list[sen]] == 0
        sen_ids_list[sentence_list[sen]] = getid(ivocab_trg, sen)

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
    '''
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


def splitline(line):
    result = line.split(' ')
    pos = 0
    while pos < len(result):
        #if result[pos][0] == '<' and result[pos][-1] != '>':
        #if result[pos][0] == '<' and not result[pos].endswith('@@') and len(result[pos]) > 1 and result[pos][-1] != '>':
        if result[pos][0] == '<' and len(result[pos]) > 1 and result[pos][-1] != '>':
            result[pos] += ' '+result[pos+1]
            del result[pos+1]
        else:
            pos += 1
    return result


def transform_gold(phrase):
    words = phrase.split(' ')
    for i in range(len(words)-1):
        if not words[i].endswith('@@'):
            words[i] = words[i] + '@@'
    return ' '.join(words)


cdef to_ascii(phrases):
    result = []
    cdef char *tmp
    cdef float prob
    for p in phrases:
        try:
            tmp = p[0]
            prob = p[1]
            result.append([tmp, prob])
        except:
            #print('dropped:', p[0].encode('utf-8'))
            continue 
    return result


cdef void print_stack(translation_status *stack, int len_src, int count):
    cdef int i
    cdef int j
    for i in range(count):
        printf('Number %d:\n', i)
        printf('translation: %s\n', stack[i].translation)
        printf('limited: [%d %d]\n', stack[i].limited[0], stack[i].limited[1])
        printf('loss: %f\n\n', stack[i].loss)


def print_element(element):
    print('translation:',element['translation'])
    print('satisfied:', element['satisfied'])
    print('num_satisfied:', sum(element['satisfied']))
    print('num_constraints:', element['num_constraints'])
    print('limited:',element['limited'])
    print('loss:', element['loss'])


cdef void print_stack_finished(translation_status *stack, int count):
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


def getid_nosplit(ivocab, text):
    if text == '':
        return [ivocab['<eos>']]
    words = text
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
    for i in cov:
        if i != 1:
            return False
    return True


def load_null2trg(n2tfile):
    content = open(n2tfile, 'r').read()
    result = content.split('\n')
    if result[-1] == '':
        del result[-1]
    result = [[tmp.split(' ')[0], float(tmp.split(' ')[1])] for tmp in result]
    return result


def load_line(filename):
    content = open(filename, 'r').read()
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



def sorted_index(inp):
    tosort = []
    for i in range(len(inp)):
        tosort.append([inp[i], i])
    tosort = sorted(tosort, key=lambda x: x[0], reverse=True)
    result = [i[1] for i in tosort]
    return result
   

def reverse_bpe(inp):
    return inp.replace('@@ ', '')

def reverse_bpe_hard(inp):
    return inp.replace('@@ ', '').replace('@@', '')


cdef int is_equal(float a, float b):
    if abs(a-b) < 1e-6:
        return 1
    return 0


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

cdef float get_lp(int length, float alpha):
    return pow((5.0 + length) / 6.0, alpha)


time_totalsp = 0

def parse_input(input):
    contents = input.split('\t')
    src = contents[0]
    constraints = []
    for i in range(1,len(contents)):
        constraints.append(contents[i])
    return src, constraints

def get_index_cons(constraints, ivocab_trg):
    result = []
    for cons in constraints:
        tmp = []
        words = cons.split()
        for w in words:
            tmp.append(ivocab_trg[w])
        result.append(tmp)
    return result

cdef float my_log(float x):
    if x == 0:
        return -10000
    else:
        return log(x)

cpdef main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()
    print("start")

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.checkpoint, args.model, params)
    override_parameters(params, args)

    ### C version params ###
    #cdef params_s params_c
    #params_c.beam_size = params.beam_size
    #params_c.keep_status_num = params.keep_status_num
    #params_c.split_limited = params.split_limited
    #params_c.src2null_loss = params.src2null_loss
    #params_c.bpe_phrase = params.bpe_phrase
    ### declaration for Cython ###
    cdef:
        # params
        int decode_length = params.decode_length
        float decode_alpha = params.decode_alpha
        int beam_size = params.beam_size
        # universal
        int max_len_trg = 150
        int max_len_src = 150
        int max_candidate = 5000
        int i, j, k, len_tmp, pos, pos_end, offset
        char *tmpstring
        #phrase_pair *phrases_c
        int best_count = 0
        float threshold = -1000000
        # encode & prepare
        float time_test[20]
        int count_test[20]
        int len_src, length
        translation_status element_init
        translation_status stacks[150][150][4]
        int stacks_count[150][150]
        int finished_count
        # src2null
        int num_cov
        int all_empty
        translation_status element
        translation_status newelement
        float new_loss, new_word_loss
        float new_align_loss
        # neural
        int maps[150][4]
        int num_x
        # generation
        int len_covered, nosense, is_visible, num_total, tmp_id, have_first, total_length, len_first
        float prob_align, total_src2null_loss
        beam add_result
        char *tmpstr2
        char *newbuffer
        char *firstword
        char *firstword2
        int candidate_phrase_list_count[150]
        float length_penalty
        float end_s2n_loss
        int current_state
    # Build Graph
    with tf.Graph().as_default():
        model = model_cls(params)
        #print('input file:', args.input)
        inputs = read_files([args.input])[0]
        #print('inputs', inputs)

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
        assign_op = tf.group(*ops)


        fd = tf.gfile.Open(args.output, "w")

        fout = open(args.output, 'w')
        count = 0

        # Create session
        sess = tf.train.MonitoredSession(session_creator=sess_creator)
        sess.run(assign_op)

        total_start = time.time()
        for input in inputs:
            try:
                count += 1
                print('count:', count)
                start = time.time()
                src, constraints = parse_input(input)
                constraints_idx = get_index_cons(constraints, ivocab_trg)
                num_constraints = len(constraints)
                len_constraints = [len(constraints_idx[i]) for i in range(len(constraints_idx))]
                real_beam_size = params.beam_size
                src = copy.deepcopy(src)
                src = src.decode('utf-8')

                if args.verbose:
                    print('constraints:', constraints)
                    print('constraints_id:', constraints_idx)
                    print('constraints_len:', len_constraints)

                #words = splitline(src)
                words = src.split()

                len_src = len(words)
                #print("len_src:", len_src)
                f_src = {}
                #f_src["source"] = [getid(ivocab_src, input) ]
                f_src["source"] = [getid_nosplit(ivocab_src, [w.encode('utf-8') for w in words]) ]
                f_src["source_length"] = [len(f_src["source"][0])] 
                #print('input_enc', f_src)
                feed_src = {
                    placeholder["source"]: f_src["source"],
                    placeholder["source_length"] : f_src["source_length"]
                }
                encoder_state = sess.run(enc, feed_dict=feed_src)


                print('source:', src.encode('utf-8'))

                state_init = {}
                state_init["encoder"] = encoder_state 
                for i in range(params.num_decoder_layers):
                    state_init["decoder"] = {}
                    state_init["decoder"]["layer_%d" % i] = np.zeros((0, params.hidden_size))

                '''
                element_init.translation = ''
                element_init.hidden_state_id = 0
                element_init.limited = [-1, -1]
                element_init.satisfied = [0]*20
                hidden_state_pool = [getstate(encoder_state, params.num_decoder_layers)]
                element_init.loss = 0
                '''
                element_ini = {}
                element_ini['translation'] = ''
                element_ini['limited'] = []
                element_ini['satisfied'] = [0]*20
                element_ini['hidden'] = getstate(encoder_state, params.num_decoder_layers)
                element_ini['num_constraints'] = 0
                element_ini['loss'] = 0
                stack = [element_ini]
                #stacks[0][0][0] = element_init
                #stacks_count = [[0]*150]*150
                #stacks_count[0][0] = 1

                beam_finished = []
                finished_count = 0
                timestep = 0

                while True:

                    # do the neural
                    if args.verbose:
                        print('timestep:', timestep)
                        for ele in stack:
                            print_element(ele)
                            print()
                    if args.verbose:
                        printf('start neural prepare\n')
                    #features, num_x = get_feature_map(stacks[length], stacks_count[length], num_constraints, ivocab_trg, hidden_state_pool, params, maps, threshold)
                    sentences = [ele['translation'] for ele in stack]
                    hidden_states = [ele['hidden'] for ele in stack]
                    sen_ids_list = [0] * len(sentences) 
                    sentence_num = len(sentences)
                    num_x = sentence_num
                    for i in range(len(sentences)):
                        sen = sentences[i]
                        sen_ids_list[i] = getid(ivocab_trg, sen)
                    max_len = max(map(len, sen_ids_list))
                    #print(sen_ids_list)
                    padded_input = np.ones([sentence_num, max_len], dtype=np.int32) * ivocab_trg['<pad>']
                    for i in range(sentence_num):
                        padded_input[i][:len(sen_ids_list[i])] = sen_ids_list[i]
                    features = {}
                    features["target"] = padded_input
                    features["target_length"] = [len(sen_ids) for sen_ids in sen_ids_list]
                    features["decoder"] = {}
                    for i in range(params.num_decoder_layers):
                        # the main timecost is here
                        features["decoder"]["layer_%d" % i] = merge_tensor(hidden_states, i)
                    features["encoder"] = np.tile(encoder_state["encoder"], (num_x, 1, 1))
                    features["source"] = [getid_nosplit(ivocab_src, [w.encode('utf-8') for w in words]) ] * num_x 
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

                    #printf('start neural\n')
                    # log probs: [sen_num, vocab_trg]
                    log_probs, new_state = sess.run(dec, feed_dict=feed_dict)
                    #print('tmp', type(log_probs), log_probs.shape)
                    new_state = outdims(new_state, params.num_decoder_layers)
                    '''
                    new_state = outdims(new_state, params.num_decoder_layers)
                    hidden_state_pool = new_state
                    for num_cov in range(0, len_src+1):
                        if stacks_count[length][num_cov] == 0:
                            continue
                        neural_result[num_cov] = [[], []]
                        for i in range(stacks_count[length][num_cov]):
                            pos = maps[num_cov][i]
                            if pos == -1:
                                neural_result[num_cov][0].append(-1)
                                neural_result[num_cov][1].append(-1)
                            else:
                                neural_result[num_cov][0].append(log_probs[pos])
                                neural_result[num_cov][1].append(pos)
                    '''

                    candidates = []
                    # generate next
                    # global k-best
                    log_probs_re = np.reshape(log_probs, [log_probs.shape[0]*len(ivocab_trg)])
                    log_probs_eos = [0]*real_beam_size
                    # disable generation of <eos>
                    for i in range(len(stack)):
                        log_probs_eos[i] = log_probs[i][1]
                        log_probs_re[i*len(ivocab_trg)+1] = -10000
                    index_kbest = np.argpartition(log_probs_re, -real_beam_size)[-real_beam_size:]
                    loss = log_probs_re[index_kbest]
                    #print('kbest:', index_kbest)
                    beam_indices = index_kbest // len(ivocab_trg)
                    symbol_indics = index_kbest % len(ivocab_trg)
                    loss_origin = [stack[bi]['loss'] for bi in beam_indices]
                    loss = loss+loss_origin
                    #print('indices', zip(beam_indices, symbol_indics, loss))
                    candidates += zip(beam_indices, symbol_indics, loss)

                    # constraint
                    #print('constraints:', constraints, constraints_idx)
                    for i in range(len(stack)):
                        ele = stack[i] 
                        for j in range(len(constraints)):
                            if ele['satisfied'][j] == 1:
                                continue
                            inlimit = False
                            for limit_index, limit_count in ele['limited']:
                                if j == limit_index:
                                    newword = constraints_idx[j][limit_count+1]
                                    inlimit = True
                                    break
                            if not inlimit:
                                newword = constraints_idx[j][0]
                            loss = ele['loss']+log_probs[i][newword]
                            candidates.append((i, newword, loss))
                                
                    #print('cand:', candidates)

                    # single-best for each hypo
                    index_sb = np.argpartition(log_probs, -1)
                    #print('index sb:', index_sb)
                    for i in range(len(stack)):
                        single_best = index_sb[i][-1]
                        loss = stack[i]['loss']+log_probs[i][single_best]
                        candidates.append((i, single_best, loss))

                    candidates = list(set(candidates))
                    #print('final cand:', candidates)
                    
                    # judge constraints 
                    have_ele = [0]*50
                    new_eles = []
                    for cand in candidates:
                        ele = stack[cand[0]]
                        #print('cand:', cand)
                        # <eos>
                        if sum(ele['satisfied']) == len(constraints):
                            loss_end = (ele['loss']+log_probs_eos[cand[0]]) / get_lp(timestep, params.decode_alpha)
                            pos = len(beam_finished)
                            while pos > 0:
                                if beam_finished[pos-1][1] < loss_end:
                                    pos -= 1
                                else:
                                    break
                            beam_finished.insert(pos, [ele['translation'], loss_end])
                            if len(beam_finished) > real_beam_size:
                                del beam_finished[-1]
                        new_ele = {}
                        new_ele['translation'] = ele['translation']+' '+params.vocabulary["target"][cand[1]]
                        new_ele['translation'] = new_ele['translation'].strip()
                        new_satis = copy.deepcopy(ele['satisfied'])
                        new_limited = []
                        for j in range(len(constraints)):
                            finished = False
                            if ele['satisfied'][j]:
                                continue
                            for limit_index, limit_count in ele['limited']:
                                inlimit = False
                                if j == limit_index:
                                    if cand[1] == constraints_idx[limit_index][limit_count+1]:
                                        if len(constraints_idx[j]) == limit_count+2:
                                            new_satis[j] = 1
                                            finished = True
                                            break
                                        else:
                                            new_limited.append([j, limit_count+1])
                            if finished:
                                break
                            if not inlimit and constraints_idx[j][0] == cand[1]:
                                if len(constraints_idx[j]) == 1:
                                    new_satis[j] = 1
                                    new_limited = []
                                    break
                                else:
                                    new_limited.append([j, 0])
                        new_ele['limited'] = new_limited
                        new_ele['satisfied'] = new_satis
                        new_ele['loss'] = cand[2]
                        new_ele['hidden'] = new_state[cand[0]]
                        ntmp = 0
                        for j in range(len(len_constraints)):
                            if new_ele['satisfied'][j] == 1:
                                ntmp += len_constraints[j]
                        if len(new_ele['limited']) > 0:
                            ntmp += new_ele['limited'][0][1]+1
                        have_ele[ntmp] = 1
                        new_ele['num_constraints'] = ntmp
                        new_eles.append(new_ele)
                        #print_element(new_ele)
                        
                    # allocate beam
                    num_have = sum(have_ele)
                    num_each = real_beam_size // num_have
                    num_high = real_beam_size % num_have

                    allocation = [0] * 50
                    highest = -1
                    for j in range(len(have_ele)):
                        if have_ele[j] == 1:
                            allocation[j] = num_each
                            highest = j
                    allocation[highest] += num_high
                    
                            
                    ele_by_cons = []
                    new_stack = []
                    for j in range(len(allocation)):
                        ele_by_cons.append([])
                    for ele in new_eles:
                        ele_by_cons[ele['num_constraints']].append(ele)
                    if args.verbose:
                        print('allocation:', allocation)
                        print('real:', [len(ele_by_cons[j]) for j in range(len(ele_by_cons))])

                    for j in range(len(allocation)):
                        if allocation[j] > 0:
                            if allocation[j] >len(ele_by_cons[j]):
                                new_stack += ele_by_cons[j]
                                continue
                            losses = [ele['loss'] for ele in ele_by_cons[j]]
                            losses = np.array(losses)
                            loss_topk_indices = np.argpartition(losses, -allocation[j])[-allocation[j]:]
                            for index in loss_topk_indices:
                                new_stack.append(ele_by_cons[j][index])
                    if args.verbose:
                        print('finished:', len(beam_finished))
                        print('==finished==')
                        for i in range(len(beam_finished)):
                            print('translation:', beam_finished[i][0])
                            print('loss:', beam_finished[i][1])
                            print()

                    timestep += 1
                    stack = new_stack
                    if timestep > len(words)+params.decode_length:
                        break
                    #exit()
                    


                if len(beam_finished) > 0:
                    result = beam_finished[0][0]
                else: 
                    result = ''
            except:
                result = 'FAILED'
                traceback.print_exc()


            if result == "":
                fout.write('\n')
                continue
            try:
                fout.write((result.replace(' <eos>', '').strip()+'\n').encode('utf-8'))
            except:
                fout.write((result.replace(' <eos>', '').strip()+'\n'))
                
            try:
                print((result.replace(' <eos>', '').strip()).encode('utf-8'))
            except:
                print((result.replace(' <eos>', '').strip()))
                #print("print result error")

            end = time.time()
            global time_totalsp

            #try:
            #    print('final:', ' '.join(words_trg).encode('utf-8'))
            #    fout.write((' '.join(' '.join(words_trg).split()).strip()+'\n').encode('utf-8'))
            #except:
            #    print('print final failed')
            #    print(' '.join(words_trg))
            #    fout.write((' '.join(' '.join(words_trg).split()).strip()+'\n'))

            end = time.time()
            print('sentence time:', end-start, 's')
        total_end = time.time()
        print('total time:', total_end-total_start, 's')


if __name__ == "__main__":
    main(parse_args())
