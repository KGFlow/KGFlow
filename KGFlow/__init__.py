# coding=utf-8
import tensorflow as _tf

if _tf.__version__[0] == "1":
    _tf.enable_eager_execution()

import KGFlow.data as data
import KGFlow.dataset as dataset
import KGFlow.utils as utils

from KGFlow.data import *
