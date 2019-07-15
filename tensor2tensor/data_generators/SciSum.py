



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import random
import tarfile
import pandas as pd
import requests
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry

import tensorflow as tf

EOS = text_encoder.EOS_ID

# Techniques for data prep from See et al. (2017)
dm_single_close_quote = u"\u2019"  # unicode
dm_double_close_quote = u"\u201d"
# Acceptable ways to end a sentence.
END_TOKENS = [
    u".", u"!", u"?", u"...", u"'", u"`", u"\"", dm_single_close_quote,
    dm_double_close_quote, u")"
]

if not os.path.exists('InitialData'):
        os.makedirs('InitialData')
        
        
