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


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--phrase", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    phrase_table = json.load(open(args.phrase, 'r'))
    fout = open(args.output, 'w')
    for k in phrase_table.keys():
        for trans in phrase_table[k]:
            fout.write((k+'\t'+trans+'\n').encode('utf-8'))
        
