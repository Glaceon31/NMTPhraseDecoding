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

    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--prob", type=str, required=True)
    parser.add_argument("--smtprob", type=str) 
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    dev = open(args.dev, 'r').read()
    prob = open(args.prob, 'r').read()
    smtprob = json.load(open(args.smtprob, 'r'))
    devlines = dev.split('\n')
    problines = prob.split('\n')
    print(smtprob[","])

    output = open(args.output, 'w')
    for i in range(len(devlines)):
        devline = devlines[i]
        if devlines == '':
            continue
        probline = problines[i]
        devwords = devline.split(' ')
        probwords = probline.split(' ')[:-1]

        if args.smtprob:
            result = [devw+'/'+probw+'('+(str(round(1-smtprob[devw.decode('utf-8')][2], 2)) if smtprob.has_key(devw.decode('utf-8')) else 'no')+')' for devw, probw in zip(devwords, probwords)]
        else:
            result = [devw+'/'+probw for devw, probw in zip(devwords, probwords)]

        resultlines = ' '.join(result)
        output.write(resultlines+'\n')


