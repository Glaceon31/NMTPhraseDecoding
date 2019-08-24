from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import time
import math
import json
import re


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    input = open(args.input, 'r').read()
    output = open(args.output, 'w')
    constraints = re.finditer(r'<cons translation=".*?"> .*? </cons>', input)
    result = ""
    pos = 0
    for c in constraints:
        cons = input[c.start():c.end()]
        trg, trans = re.findall(r'<cons translation="(.*?)"> (.*?) </cons>', cons)[0]
        print(trg, '|||', trans)
        if trg in trans:
            print('no replace')
            result += input[pos:c.start()]+trans
            pos = c.end()
            continue
        # pre
        if trans in trg:
            if trg in input[(c.start()+len(trg)-len(trans)):c.start()]+trans+input[c.end():(c.end()+len(trg)-len(trans))]:
                print('no replace2')
                result += input[pos:c.start()]+trans
                pos = c.end()
                continue
        #replace
        print('replace')
        result += input[pos:c.start()]+trg
        pos = c.end()
    result += input[pos:]
    output.write(result)
    output.close()
