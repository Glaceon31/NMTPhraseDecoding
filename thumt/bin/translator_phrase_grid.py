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
    parser.add_argument("--nullprob", type=str, required=True,
                        help="probability for source word to null")
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
        # phrase specific
        bpe_phrase=True,
        merge_status="max_align",
        keep_status_num=1,
        cut_ending=False,
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


def subset(phrases, words, ngram, cov=None, rbpe=False):
    result = {}
    covered = [0] * len(words)
    for i in range(len(words)):
        if cov:
            if cov[i] != 0:
                continue
        for j in range(i+1, min(i+ngram+1, len(words)+1)):
            if cov :
                if cov[j-1] != 0:
                    break
            phrase = ' '.join(words[i:j])
            if phrases.has_key(phrase):
                result[phrase] = phrases[phrase]
                for k in range(i, j):
                    covered[k] = 1
    # special treatment for words with no phrase
    for i in range(len(words)):
        if cov:
            if cov[i] != 0:
                continue
        if result.has_key(words[i]):
            if not [words[i], 1.0] in result[words[i]]:
                result[words[i]].append([words[i], 1.0])
        else:
            result[words[i]] = [[words[i], 1.0]]
            
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
        return result_rbpe
    else:
        return result


def print_phrases(phrases):
    for k in phrases.keys():
        tmp = k+' ||| '
        for t in phrases[k]:
            if type(t) is list:
                tmp += t[0]+' '+str(t[1])+' ||| '
        print(tmp.encode('utf-8'))


def print_stack(stack):
    for i in range(len(stack)):
        s = stack[i]
        print('s%d:' %i, s[0].encode('utf-8'), s[1], s[3], s[4])


def print_stack_finished(stack):
    for i in range(len(stack)):
        s = stack[i]
        print('end%d:' %i, s[0].encode('utf-8'), s[-1])


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


def find_translate(now, previous, nowst, lastst, words_src):
    if not nowst:
        return
    nowst = json.loads(nowst)
    lastst = json.loads(lastst)
    pos = -1
    for i in range(len(nowst[0])):
        if nowst[0][i] == 1 and lastst[0][i] == 0:
            pos = i
            break
    if pos > 0:
        src = words_src[pos]
        if nowst[1][0] == 'normal':
            trg = now[0].split(' ')[-1]
        elif nowst[1][0] == 'limited':
            print('limit:', nowst)
            trg = now[0].split(' ')[-1]+' '+' '.join(nowst[1][1])
            
        print('translation:', src.encode('utf-8'), '-', trg)
    return 


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


def is_finish(element):
    cov = element[0]
    status = element[1]
    if status[0] == 'limited':
        return False
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


def append_empty(stacks, length, cov_num):
    while len(stacks) <= length:
        stacks.append([])
    while len(stacks[length]) <= cov_num:
        stacks[length].append([])
    return stacks


def can_addstack(stack, loss, beam_size, align_loss=None):
    if len(stack) < beam_size:
        return True
    else:
        if loss >= stack[-1][-1]:
            if align_loss:
                for s in stack:
                    if loss == s[-1]:
                        # pending: consider multiple status
                        if align_loss > json.loads(s[1].keys()[0])[2]:
                            return True
            else:
                return True
    return False


def argmax(array, avail, num):
    tosort = []
    for i in range(len(array)):
        if avail[i] == 0:
            tosort.append([array[i], i])
    tosort = sorted(tosort, key=lambda x: x[0], reverse=True)
    return [i[1] for i in tosort]
    

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
    return tmp[2]


def add_stack(stack, element, beam_size, merge_status=None, max_status=1):
    if merge_status:
        for i in range(len(stack)):
            if element[0] == stack[i][0]: 
                st = element[1].keys()[0]
                if not stack[i][1].has_key(st):
                    if merge_status == "keep_all":
                        stack[i][1][st] = 1
                    elif merge_status == "max_align":
                        if len(stack[i][1]) < max_status:
                            stack[i][1][st] = 1
                        else:
                            pnew = get_align_prob(element[1].keys()[0])
                            pmin, whichmin = min_align_prob(stack[i][1])
                            if pnew > pmin:
                                stack[i][1][st] = 1
                                del stack[i][1][whichmin]
                return stack
    else:
        for i in range(len(stack)):
            if element[0] == stack[i][0]: 
                return stack
    if len(stack) < beam_size:
        result = stack + [element]
    else:
        if element[-1] > stack[-1][-1]:
            result = stack[:beam_size-1] + [element]
        else:
            return stack
    result = sorted(result, key=lambda x:x[-1], reverse=True)
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


time_totalsp = 0
def update_stack(stack_current, finished, log_probs, new_state, words_src, phrases, phrases_reverse, ivocab_trg, params):
    stack_new = []
    finished_new = finished
    time_start = time.time()
    global time_totalsp
    time_sp = 0
    count_all = 0
    count_canadd = 0
    for i in range(len(stack_current)):
        tmp = stack_current[i]
        for s in range(len(tmp[1].keys())):
            status_str = tmp[1].keys()[s]
            status = json.loads(status_str)
            # limit
            if status[1][0] == "limited":
                limits = status[1][1]
                count_all += 1
                new_loss = float(tmp[-1]+log_probs[i][getid_word(ivocab_trg, limits[0])])
                if not can_addstack(stack_new, new_loss, params.beam_size):
                    continue
                count_canadd += 1
                newstatus = copy.deepcopy(status)
                if len(limits) == 1:
                    newstatus[1][0] = "normal"
                    newstatus[1][1] = ""
                else:
                    newstatus[1][1] = limits[1:]
                new = [(tmp[0]+' '+limits[0]).strip(), {json.dumps(newstatus):1}, new_state[i], [i, status_str], new_loss]
                stack_new = add_stack(stack_new, new, params.beam_size, params.merge_status, params.keep_status_num)
            else:
                all_covered = True
                for pos in range(len(status[0])):
                    # bpe_phrase
                    if params.bpe_phrase:
                        pos_end = pos
                        while pos < len(words_src) and status[0][pos] == 0 and words_src[pos_end].endswith('@@'):
                            pos_end += 1
                        if pos_end > pos:
                            bpe_phrase = ' '.join(words_src[pos:pos_end+1])
                            if phrases.has_key(bpe_phrase):
                                for j in range(len(phrases[bpe_phrase])):
                                    if not type(phrases[bpe_phrase][j]) is list:
                                        continue
                                    count_all += 1
                                    phrase, prob_align = phrases[bpe_phrase][j]
                                    words = phrase.split(' ')
                                    new_loss = float(tmp[-1]+log_probs[i][getid_word(ivocab_trg, words[0])])
                                    if not can_addstack(stack_new, new_loss, params.beam_size):
                                        continue
                                    count_canadd += 1
                                    newstatus = copy.deepcopy(status)
                                    for k in range(pos, pos_end+1):
                                        newstatus[0][k] = 1
                                    if len(words) > 1:
                                        newstatus[1] = ["limited", words[1:]]
                                    else:
                                        newstatus[1] = ["normal", ""]
                                    newstatus[2] += math.log(prob_align)
                                    new = [(tmp[0]+' '+words[0]).strip(), {json.dumps(newstatus):1}, new_state[i], [i, status_str], new_loss]
                                    stack_new = add_stack(stack_new, new, params.beam_size, params.merge_status, params.keep_status_num)
                                    
                    # generate from source word
                    if status[0][pos] == 0:
                        all_covered = False
                        num_total = len(phrases[words_src[pos]])
                        time_sps = time.time()
                        for j in range(num_total):
                            if not type(phrases[words_src[pos]][j]) is list:
                                continue
                            count_all += 1
                            phrase, prob_align = phrases[words_src[pos]][j]
                            words = phrase.split(' ')
                            new_loss = float(tmp[-1]+log_probs[i][getid_word(ivocab_trg, words[0])])
                            if not can_addstack(stack_new, new_loss, params.beam_size):
                                continue
                            count_canadd += 1
                            newstatus = copy.deepcopy(status)
                            newstatus[0][pos] = 1
                            if len(words) > 1:
                                newstatus[1] = ["limited", words[1:]]
                            else:
                                newstatus[1] = ["normal", ""]
                            newstatus[2] += math.log(prob_align)
                            new = [(tmp[0]+' '+words[0]).strip(), {json.dumps(newstatus):1}, new_state[i], [i, status_str], new_loss]
                            stack_new = add_stack(stack_new, new, params.beam_size, params.merge_status, params.keep_status_num)
                        time_spe = time.time()
                        time_sp += time_spe-time_sps
                if all_covered:
                    new_loss = float(tmp[-1]+log_probs[i][getid_word(ivocab_trg, '<eos>')])
                    new = [(tmp[0]+' <eos>').strip(), {json.dumps(status):1}, new_state[i], i, new_loss]
                    finished_new = add_stack(finished_new, to_finish(new, params.decode_alpha), params.beam_size)
     
    '''
    print('time_sp:', time_sp, 's')
    time_totalsp += time_sp
    time_end = time.time()
    print('time_update', time_end-time_start, 's')
    print('count:', count_canadd, '/', count_all)
    '''
    return stack_new, finished_new

def to_finish(state, alpha):
    result = []
    result.append(state[0])
    length = len(state[0].split(' '))
    length_penalty = math.pow((5.0 + length) / 6.0, alpha)
    length_penalty = 1.0
    result.append(state[3])
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
        #if args.tmpphrase and os.path.exists(args.tmpphrase):
        #    print('load tmpphrase')
        #    phrase_table = json.load(open(args.tmpphrase, 'r'))
        #else:
        phrase_table = json.load(open(args.phrase, 'r'))
        src2null_prob = json.load(open(args.nullprob ,'r'))

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
            count += 1
            print(count)
            start = time.time()
            src = copy.deepcopy(input)
            src = src.decode('utf-8')
            words = src.split(' ')
            len_src = len(words)
            probs_null = [src2null_prob[w][2] for w in words]
            f_src = {}
            f_src["source"] = [getid(ivocab_src, input) ]
            f_src["source_length"] = [len(f_src["source"][0])] 
            #print('input_enc', f_src)
            feed_src = {
                placeholder["source"]: f_src["source"],
                placeholder["source_length"] : f_src["source_length"]
            }
            encoder_state = sess.run(enc, feed_dict=feed_src)

            # generate a subset of phrase table for current translation
            phrases = subset(phrase_table, words, args.ngram, rbpe=args.rbpe)
            phrases_reverse = reverse_phrase(phrases)
            #print('reverse phrase:', phrases_reverse)
            print('source:', src.encode('utf-8'))
            print('probs_null:', probs_null)
            if args.rbpe:
                words = reverse_bpe(src).split(' ')
                print('reverse_bpe:', reverse_bpe(src).encode('utf-8'))
            coverage = [0] * len(words)
            
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
            2. coverage status (set), [coverage, status, align_log_prob]
            status (set): ['normal',  ''] or ['limited', limited word] 
            3. hidden state ({"layer_0": [...], "layer_1": [...]})
            4. [id from last stack, last status id]
            5. score
            '''
            element_init = ['', {json.dumps([coverage, ['normal', ''], 0.]): 1}, getstate(encoder_state, params.num_decoder_layers), [0, 0], 0]
            stacks = [[[element_init]]]
            finished = []
            length = 0

            time_neural = 0
            time_update = 0
            while True:
                # source to null
                if args.verbose:
                    print('===',length,'===')
                all_empty = True
                for num_cov in range(0, len_src+1):
                    if args.verbose:
                        print('=',num_cov,'=')
                    if len(stacks[length][num_cov]) == 0:
                        continue
                    all_empty = False
                    if len(stacks[length]) < num_cov+2:
                        stacks[length].append([])
                    for element in stacks[length][num_cov]:
                        for status_str in element[1].keys():
                            status = json.loads(status_str)
                            for nullpos in argmax(probs_null, status[0], params.beam_size):
                                if status[0][nullpos] == 1:
                                    continue
                                new_align_loss = status[2]+math.log(probs_null[nullpos])
                                if not can_addstack(stacks[length][num_cov+1], element[-1], params.beam_size, align_loss=new_align_loss):
                                    continue
                                newstatus = copy.deepcopy(status)
                                newstatus[0][nullpos] = 1
                                newstatus[2] = new_align_loss
                                new = [element[0], {json.dumps(newstatus):1}]+element[2:]
                                stacks[length][num_cov+1] = add_stack(stacks[length][num_cov+1], new,params.beam_size, params.merge_status, params.keep_status_num)
                    if args.verbose:
                        print_stack(stacks[length][num_cov])

                    #if args.verbose:
                    #    print('===',length,'===')
                    #    print('=', num_cov, '=')
                    #time_a = time.time()
                    #if len(stack_current) == 0:
                    #    break
                    #stack_current = sorted(stack_current, key=lambda x: x[-1], reverse=True)
                    #stack_current = stack_current[:params.beam_size]
                    #if args.verbose:
                    #    print_stack(stack_current)
                    #    print([len(ss[1]) for ss in stack_current])
                    #time_b = time.time()

                    # do the neural
                    stack_current = stacks[length][num_cov]

                    features = get_feature(stack_current, ivocab_trg)
                    features["encoder"] = np.tile(encoder_state["encoder"], (len(stack_current), 1, 1))
                    features["source"] = [getid(ivocab_src, input)] * len(stack_current)
                    features["source_length"] = [len(features["source"][0])] * len(stack_current)
                    features["decoder"] = {}
                    for i in range(params.num_decoder_layers):
                        features["decoder"]["layer_%d" % i] = merge_tensor(stack_current, i)

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

                    time_bc = time.time()
                    #print('prepare neural:', time_bc-time_b, 's')
                    log_probs, new_state = sess.run(dec, feed_dict=feed_dict)
                    time_c = time.time()
                    #print('neural:', time_c-time_bc, 's')
                    time_neural += time_c-time_bc
                    new_state = outdims(new_state, params.num_decoder_layers)

                    #new_stack, finished = update_stack(stack_current, finished, log_probs, new_state, words, phrases, phrases_reverse, ivocab_trg, params)
                    # update the stacks
                    count_all = 0
                    count_canadd = 0
                    stacks = append_empty(stacks, length+1, len_src+1)
                    for i in range(len(stack_current)):
                        element = stack_current[i]
                        for status_str in element[1].keys():
                            status = json.loads(status_str)
                            # limit
                            if status[1][0] == "limited":
                                limits = status[1][1]
                                count_all += 1
                                new_loss = float(element[-1]+log_probs[i][getid_word(ivocab_trg, limits[0])])
                                if not can_addstack(stacks[length+1][num_cov], new_loss, params.beam_size):
                                    continue
                                newstatus = copy.deepcopy(status)
                                if len(limits) == 1:
                                    newstatus[1][0] = "normal"
                                    newstatus[1][1] = ""
                                else:
                                    newstatus[1][1] = limits[1:]
                                new = [(element[0]+' '+limits[0]).strip(), {json.dumps(newstatus):1}, new_state[i], [i, status_str], new_loss]
                                stacks[length+1][num_cov] = add_stack(stacks[length+1][num_cov], new, params.beam_size, params.merge_status, params.keep_status_num)
                            else:
                                all_covered = True
                                for pos in range(len(status[0])):
                                    # bpe_phrase
                                    if params.bpe_phrase:
                                        # find untranslated bpe phrase
                                        pos_end = pos
                                        while pos < len(words) and status[0][pos] == 0 and words[pos_end].endswith('@@'):
                                            pos_end += 1
                                        if pos_end > pos:
                                            bpe_phrase = ' '.join(words[pos:pos_end+1])
                                            len_bpe_phrase = pos_end-pos+1
                                            if phrases.has_key(bpe_phrase):
                                                # start translation
                                                for j in range(len(phrases[bpe_phrase])):
                                                    if not type(phrases[bpe_phrase][j]) is list:
                                                        continue
                                                    count_all += 1
                                                    phrase, prob_align = phrases[bpe_phrase][j]
                                                    words_p = phrase.split(' ')
                                                    new_loss = float(element[-1]+log_probs[i][getid_word(ivocab_trg, words_p[0])])
                                                    if not can_addstack(stacks[length+1][num_cov+len_bpe_phrase], new_loss, params.beam_size):
                                                        continue
                                                    count_canadd += 1
                                                    newstatus = copy.deepcopy(status)
                                                    for k in range(pos, pos_end+1):
                                                        newstatus[0][k] = 1
                                                    if len(words_p) > 1:
                                                        newstatus[1] = ["limited", words_p[1:]]
                                                    else:
                                                        newstatus[1] = ["normal", ""]
                                                    newstatus[2] += math.log(prob_align)
                                                    new = [(element[0]+' '+words_p[0]).strip(), {json.dumps(newstatus):1}, new_state[i], [i, status_str], new_loss]
                                                    stacks[length+1][num_cov+len_bpe_phrase] = add_stack(stacks[length+1][num_cov+len_bpe_phrase], new, params.beam_size, params.merge_status, params.keep_status_num)
                                    #generate from source word
                                    if status[0][pos] == 0:
                                        all_covered = False
                                        num_total = len(phrases[words[pos]])
                                        time_sps = time.time()
                                        for j in range(num_total):
                                            if not type(phrases[words[pos]][j]) is list:
                                                continue
                                            count_all += 1
                                            phrase, prob_align = phrases[words[pos]][j]
                                            words_p = phrase.split(' ')
                                            new_loss = float(element[-1]+log_probs[i][getid_word(ivocab_trg, words_p[0])])
                                            if not can_addstack(stacks[length+1][num_cov+1], new_loss, params.beam_size):
                                                continue
                                            count_canadd += 1
                                            newstatus = copy.deepcopy(status)
                                            newstatus[0][pos] = 1
                                            if len(words_p) > 1:
                                                newstatus[1] = ["limited", words_p[1:]]
                                            else:
                                                newstatus[1] = ["normal", ""]
                                            newstatus[2] += math.log(prob_align)
                                            new = [(element[0]+' '+words_p[0]).strip(), {json.dumps(newstatus):1}, new_state[i], [i, status_str], new_loss]
                                            stacks[length+1][num_cov+1] = add_stack(stacks[length+1][num_cov+1], new, params.beam_size, params.merge_status, params.keep_status_num)
                                if all_covered:
                                    print('all covered')
                                    new_loss = float(element[-1]+log_probs[i][getid_word(ivocab_trg, '<eos>')])
                                    new = [(element[0]+' <eos>').strip(), {json.dumps(status):1}, new_state[i], i, new_loss]
                                    finished = add_stack(finished, to_finish(new, params.decode_alpha), params.beam_size)

                    time_d = time.time()
                    #print('update stack:', time_d-time_c, 's')
                    time_update += time_d-time_c
                    #finished = sorted(finished, key=lambda x: x[-1], reverse=True)
                    #finished = finished[:params.beam_size]
                    #stack_current = new_stack
                    #stacks.append(stack_current)
                    if args.verbose:
                        print_stack_finished(finished)
                if all_empty:
                    break
                length += 1
                #exit()
                #if length > 1:
                #    exit()
            print('neural:', time_neural, 's')
            print('total update stack:', time_update, 's')

            if params.cut_ending:
                len_max = len(finished[0][0].split(' '))-1
                len_now = len_max
                loss = [0] * (len_now+1)
                pos_cut = len_now+1
                now = stacks[len_now][finished[0][1]]
                while len_now > 0:
                    loss[len_now] = now[-1]
                    now = stacks[len_now-1][now[3][0]]
                    len_now -= 1
                for p in range(15, len_max+1):
                    cut = True
                    sum_ = 0.
                    for tp in range(p, len_max+1):
                        sum_ = loss[p-1]-loss[tp]
                        if sum_/(tp-p+1) < params.cut_threshold:
                            cut = False
                            break
                    if cut:
                        pos_cut = p
                        break
                result = ' '.join(finished[0][0].split(' ')[:pos_cut-1])

                print('loss:', loss)
            else:
                result = finished[0][0]

            #fout.write((finished[0][0].replace(' <eos>', '').strip()+'\n').encode('utf-8'))
            fout.write((result.replace(' <eos>', '').strip()+'\n').encode('utf-8'))
            print((result.replace(' <eos>', '').strip()).encode('utf-8'))
            print((finished[0][0].replace(' <eos>', '').strip()).encode('utf-8'))

            end = time.time()
            global time_totalsp
            if args.time:
                print('time total sp:', time_totalsp, 's')
                print('time:', end-start, 'seconds')

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

if __name__ == "__main__":
    main(parse_args())
