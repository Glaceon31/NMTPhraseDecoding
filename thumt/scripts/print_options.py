#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os
import json

import numpy as np
import tensorflow as tf


def parseargs():
    msg = "Average checkpoints"
    usage = "average.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    options = json.load(open(args.input, 'r'))
    output = open(args.output, 'w')
    for k in options.keys():
        tmp = k+' ||| '
        for t in options[k]:
            if type(t) is list:
                tmp += t[0]+' '+str(t[1])+' ||| '
        output.write(tmp.encode('utf-8')+'\n')
