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


def parseargs():
    msg = "sort null2trg probability"
    usage = "sort_null2trg.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--prob", type=str, required=True,
                        help="probability file")
    parser.add_argument("--output", type=str, help="output path")
    parser.add_argument("--mincount", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    probs = json.load(open(args.prob, 'r')) 

    prob_sorted = sorted(probs.items(), key=lambda x:x[1][2], reverse=True)
    output = open(args.output, 'w')
    for tmp in prob_sorted:
        if tmp[1][1] < args.mincount:
            continue
        output.write((tmp[0]+" "+str(tmp[1][2])+"\n").encode('utf-8'))
