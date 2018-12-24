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
import math


def parseargs():
    msg = "Merge translation options"
    usage = "merge_translation_option.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="translation options")
    parser.add_argument("--output", type=str, help="output path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    result = {} 
    tos = [json.load(open(i, 'r')) for i in args.input]
    num_options = len(args.input)
    for i in range(len(tos)):
        print('option', i, ':', len(tos[i]), 'phrases')
    for i in range(len(tos)):
        for key in tos[i].keys():
            if result.has_key(key):
                continue
            tmp_options = {}
            for j in range(len(tos)):
                if tos[j].has_key(key):
                    for item in tos[j][key]:
                        if tmp_options.has_key(item[0]):
                            tmp_options[item[0]] += item[1]
                        else:
                            tmp_options[item[0]] = item[1]
            tmp_options = [list(k) for k in tmp_options.items()]
            tmp_options = [[k[0], k[1]/num_options] for k in tmp_options]
            result[key] = tmp_options
            if len(result) % 10000 == 0:
                print(len(result))
                for j in range(len(tos)):
                    if tos[j].has_key(key):
                        print(tos[j][key])
                print(tmp_options)
    print('total:', len(result))
    json.dump(result ,open(args.output, 'w'))
                    

