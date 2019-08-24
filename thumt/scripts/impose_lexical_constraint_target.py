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
    output.write(re.sub(r'<cons translation="(.*?)"> .*? </cons>', r'\1', input))
    output.close()
