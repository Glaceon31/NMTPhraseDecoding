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

import numpy as np
import math
import time

puncfile = '/data/zjctmp/ACL2019/exp/puncs'

def load_punc(puncfile):
    content = open(puncfile, 'r').read()
    result = content.split('\n')
    if result[-1] == '':
        del result[-1]
    return result

punc = load_punc(puncfile)

def build(words_src, params):
    result = {'states': [], 'init_state': 0}
    length = len(words_src)
    if params.punc_border:
        start = 0
        for i in range(length):
            if words_src[i] in punc:
                state = {}
                state['visible'] = range(start, i+1)
                state['next_state'] = None
                result['states'].append(state)
                start = i+1
        if start < length:
            state = {}
            state['visible'] = range(start, length)
            state['next_state'] = None
            result['states'].append(state)
        for i in range(len(result['states'])-1):
            result['states'][i]['next_state'] = i+1
    else:
        state = {}
        state['visible'] = range(length)
        state['next_state'] = None
        result['states'].append(state)
                
    return result


def can_go_next(state, coverage):
    for i in range(len(state['visible'])):
        if coverage[state['visible'][i]] == 0:
            return False
    return True


def print_autom(autom):
    print(autom)
    return
