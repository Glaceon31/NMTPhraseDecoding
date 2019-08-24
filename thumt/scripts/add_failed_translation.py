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
    msg = "add failed translation"
    usage = "add_failed_translation.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)
    parser.add_argument("--hypo", type=str, required=True)
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    hypo = open(args.hypo, 'r').read()
    base = open(args.base, 'r').read()
    result = []
    lines_hypo = hypo.split('\n')
    lines_base = base.split('\n')
    if lines_hypo[-1] == '':
        del lines_hypo[-1]
    for i in range(len(lines_hypo)):
        h = lines_hypo[i]
        if h != '':
            result.append(h)
        else:
            result.append(lines_base[i])
    output = open(args.output, 'w')
    output.write('\n'.join(result)+'\n')
