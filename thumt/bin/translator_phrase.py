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
    parser.add_argument("--ngram", type=int, default=4,
                        help="ngram length")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")

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
        decode_alpha=0.6
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
            
        #if covered[i] == 0:
        #    result[words[i]] = [words[i]]
    return result


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


def getid_word(ivocab, word):
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


def is_finish(cov):
    for i in cov:
        if i != 1:
            return False
    return True


def getstate(encoderstate, num_layers):
    result = {"layer_%d" % i: encoderstate["decoder"]["layer_%d" % i]
              for i in range(num_layers)}
    return result


def merge_tensor(stack, layer):
    result = {}
    result["key"] = np.concatenate([stack[i][2]["layer_%d" % layer]["key"] for i in range(len(stack))])
    result["value"] = np.concatenate([stack[i][2]["layer_%d" % layer]["value"] for i in range(len(stack))])
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


def update_stack(stack_current, finished, log_probs, new_state, words_src, phrases, ivocab_trg, alpha):
    stack_new = []
    finished_new = finished
    for i in range(len(stack_current)):
        tmp = stack_current[i]
        for status in tmp[1]:
            if status[1][0] == "limited":
                limits = status[1][1]
                newstatus = copy.deepcopy(status)
                if len(limits) == 1:
                    for j in range(status[1][-1][0], status[1][-1][1]):
                        newstatus[0][j] = 1
                    newstatus[1] = ["normal", "", ""]
                else:
                    newstatus[1][1] = limits[1:]
                new = [(tmp[0]+' '+limits[0]).strip(), newstatus, new_state[i], i, float(tmp[-1]+log_probs[i][getid_word(ivocab_trg, limits[0])])]
                if is_finish(new[1][0]):
                    finished_new.append(to_finish(new, alpha))
                else:
                    stack_new.append(new)
            else:
                for k in phrases.keys():
                    found = find(words_src, status[0], k)
                    if found != -1:
                        for p in phrases[k]:
                            newstatus = copy.deepcopy(status)
                            words_trg = p.split(' ')
                            assert len(words_trg) >= 1
                            #print('newstate', new_state)
                            if len(words_trg) > 1:
                                newstatus[1] = ["limited", words_trg[1:], found]
                            else:
                                for j in range(found[0], found[1]):
                                    newstatus[0][j] = 1
                            new = [(tmp[0]+' '+words_trg[0]).strip(), newstatus, new_state[i], i, float(tmp[-1]+log_probs[i][getid_word(ivocab_trg, words_trg[0])])]
                            if is_finish(new[1][0]): 
                                finished_new.append(to_finish(new, alpha))
                            else:
                                stack_new.append(new)
     
    return stack_new, finished_new

def to_finish(state, alpha):
    result = []
    result.append(state[0])
    length = len(state[0].split(' '))
    length_penalty = math.pow((5.0 + length) / 6.0, alpha)
    result.append(state[-1] / length_penalty)
    return result

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

    # Build Graph
    with tf.Graph().as_default():
        model = model_cls(params)
        #print('input file:', args.input)
        inputs = read_files([args.input])[0]
        #print('inputs', inputs)

        # Load phrase table
        phrase_table = json.load(open(args.phrase, 'r'))

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
        #print('phrases_all:', phrase_table)

        fout = open(args.output, 'w')
        count = 0

        for input in inputs:
            count += 1
            print(count)
            start = time.time()
            src = copy.deepcopy(input)
            src = src.decode('utf-8')
            words = src.split(' ')
            f_src = {}
            f_src["source"] = [getid(ivocab_src, input) +[ivocab_src['<eos>']]]
            f_src["source_length"] = [len(f_src["source"][0])] 
            #print('input_enc', f_src)
            feed_src = {
                placeholder["source"]: f_src["source"],
                placeholder["source_length"] : f_src["source_length"]
            }
            encoder_state = sess.run(enc, feed_dict=feed_src)

            coverage = [0] * len(words)
            # generate a subset of phrase table for current translation
            phrases = subset(phrase_table, words, args.ngram)
            #print('src:', repr(src))
            #for k in phrases.keys():
            #    print(k.encode('utf-8'), len(phrases[k]))

            state_init = {}
            state_init["encoder"] = encoder_state 
            for i in range(params.num_decoder_layers):
                state_init["decoder"] = {}
                state_init["decoder"]["layer_%d" % i] = np.zeros((0, params.hidden_size))
            '''
            stacks:
            1. partial translation
            2. coverage status (set), [coverage, status]
            status (set): ['normal', '', ''] or ['limited', limited word, (left, right]] 
            3. hidden state ({"layer_0": [...], "layer_1": [...]})
            4. id from last stack
            5. score
            '''
            stack_current = [['', [coverage, ['normal', '', '']], getstate(encoder_state, params.num_decoder_layers) , 0, 0]]
            finished = []
            length = 0

            while True:
                #print('===',length,'===')
                if len(stack_current) == 0:
                    break
                stack_current = merge_duplicate(stack_current)
                stack_current = sorted(stack_current, key=lambda x: x[-1], reverse=True)
                stack_current = stack_current[:params.beam_size]
                #print("new stack:", [[s[0], s[1], s[-2], s[-1]] for s in stack_current])

                if len(stack_current) == 0:
                    continue

                features = get_feature(stack_current, ivocab_trg)
                #print('encoder state size:', encoder_state)
                features["encoder"] = np.tile(encoder_state["encoder"], (len(stack_current), 1, 1))
                features["source"] = [getid(ivocab_src, input) +[ivocab_src['<eos>']]] * len(stack_current)
                features["source_length"] = [len(features["source"][0])] * len(stack_current)
                #print("features:", features)
                features["decoder"] = {}
                for i in range(params.num_decoder_layers):
                    features["decoder"]["layer_%d" % i] = merge_tensor(stack_current, i)

                '''
                for k in features.keys():
                    if type(features[k]) == list:
                        print(k, np.asarray(features[k]).shape)
                print("encoder", features["encoder"].shape)
                print("decoder_key", features["decoder"]["layer_0"]["key"].shape)
                '''

                feed_dict = {
                    placeholder['source']: features['source'],
                    placeholder['source_length']: features['source_length'],
                    placeholder['target']: features['target'],
                    placeholder['target_length']: features['target_length']}
                #if length >= 1:
                #    scoring = sess.run(scores, feed_dict=feed_dict)
                #    print('scores:', scoring)
                dict_tmp={state["encoder"]: features["encoder"]}
                feed_dict.update(dict_tmp)
                dict_tmp={state["decoder"]["layer_%d" % i]["key"]: features["decoder"]["layer_%d" % i]["key"] for i in range(params.num_decoder_layers)}
                feed_dict.update(dict_tmp)
                dict_tmp={state["decoder"]["layer_%d" % i]["value"]: features["decoder"]["layer_%d" % i]["value"] for i in range(params.num_decoder_layers)}
                feed_dict.update(dict_tmp)

                log_probs, new_state = sess.run(dec, feed_dict=feed_dict)
                new_state = outdims(new_state, params.num_decoder_layers)
                #print("log_probs",log_probs.shape)
                #print(log_probs[0][78], log_probs[0][406])
                #print("state", new_state)

                new_stack, finished = update_stack(stack_current, finished, log_probs, new_state, words, phrases, ivocab_trg, params.decode_alpha)
                #print("new stack:", len(new_stack), [[s[0], s[-1]] for s in new_stack])
                finished = sorted(finished, key=lambda x: x[-1], reverse=True)
                finished = finished[:params.beam_size]
                stack_current = new_stack
                length += 1

            fout.write((finished[0][0].replace(' <eos>', '').strip()+'\n').encode('utf-8'))
            print((finished[0][0].replace(' <eos>', '').strip()).encode('utf-8'))

            end = time.time()
            print('total time:',end-start, 's')
        exit()
        
        '''
        for input in inputs:
            src = copy.deepcopy(input)
            src = src.decode('utf-8')
            words = src.split(' ')
            f_src = {}
            f_src["source"] = [getid(ivocab_src, input) +[ivocab_src['<eos>']]]
            f_src["source_length"] = [len(f_src["source"][0])] 
            print(f_src)
            feed_src = {
                placeholder["source"]: f_src["source"],
                placeholder["source_length"] : f_src["source_length"]
            }
            encoder_state = sess.run(enc, feed_dict = feed_src)

            coverage = [0] * len(words)
            # generate a subset of phrase table for current translation
            phrases = subset(phrase_table, words, args.ngram)
            print('src:', repr(src))
            for k in phrases.keys():
                print(k.encode('utf-8'), len(phrases[k]))
                
            state_init = {}
            state_init["encoder"] = encoder_state 
            for i in range(params.num_decoder_layers):
                state_init["decoder"] = {}
                state_init["decoder"]["layer_%d" % i] = np.zeros((0, params.hidden_size))
            stacks = [[['', coverage, 0]]]
            finished = []
            length = 0
            while True:
                print('===',length,'===')
                if len(stacks) <= length:
                    break
                if len(stacks[length]) == 0:
                    length += 1
                    continue
                stack_current = remove_duplicate(stacks[length])
                stack_current = sorted(stack_current, key=lambda x: x[2], reverse=True)
                stack_current = stack_current[:params.beam_size]
                if len(stack_current) == 0:
                    continue
                new = generate_new(words, phrases, stack_current, length)
                new = remove_duplicate(new)
                print('num_new',len(new))
                #print(new)
                features = get_feature(new, ivocab_trg)
                features["source"] = [getid(ivocab_src, input) +[ivocab_src['<eos>']]] * len(new)
                features["source_length"] = [len(features["source"][0])] * len(new)
                #print(features)
                
                feed_dict = {placeholder['source']: features['source'],
                             placeholder['source_length']: features['source_length'],
                             placeholder['target']: features['target'],
                             placeholder['target_length']: features['target_length']}
                results = sess.run(scores, feed_dict=feed_dict)
                print("num result:", len(results))
                for i in range(len(new)):
                    while len(stacks) < new[i][2]+1:
                        stacks.append([])
                    newstatus = [new[i][0], new[i][1], results[i]]
                    if is_finish(newstatus[1]):
                        if newstatus in finished:
                            continue
                        newstatus = to_finish(newstatus, params.decode_alpha)
                        if len(finished) < params.beam_size or newstatus[2] < finished[-1][2]:
                            finished.append(newstatus)
                            finished = sorted(finished, key=lambda x: x[2], reverse=True)
                            finished = finished[:params.beam_size]
                    else:
                        stacks[new[i][2]].append(newstatus)

                length += 1
            #print(finished)
            fout.write((finished[0][0].replace(' <eos>', '').strip()+'\n').encode('utf-8'))
            print((finished[0][0].replace(' <eos>', '').strip()).encode('utf-8'))
            #exit()
        end = time.time()
        print('total time:',end-start, 's')
        '''


if __name__ == "__main__":
    main(parse_args())
