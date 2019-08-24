# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.rnnsearch_lrp
import thumt.models.transformer
import thumt.models.transformer_lrp
import thumt.models.src2null
import thumt.models.src2null_prob


def get_model(name, lrp=False):
    name = name.lower()

    if name == "rnnsearch":
        if not lrp:
            return thumt.models.rnnsearch.RNNsearch
        else:
            return thumt.models.rnnsearch_lrp.RNNsearchLRP
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        if not lrp:
            return thumt.models.transformer.Transformer
        else:
            return thumt.models.transformer_lrp.TransformerLRP
    elif name == "src2null":
        return thumt.models.src2null.Src2null
    elif name == "src2null_prob":
        return thumt.models.src2null_prob.Src2null_prob
    else:
        raise LookupError("Unknown model %s" % name)
