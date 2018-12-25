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
import thumt.utils.automatons as automatons
import numpy as np
import math
import time

import pyximport; pyximport.install()
from translator_phrase_grid_c import main, parse_args


if __name__ == "__main__":
    main(parse_args())

