# coding=utf-8
import tensorflow as __tf

if __tf.__version__[0] == "1":
    __tf.enable_eager_execution()

import KGFlow.data as data
import KGFlow.dataset as dataset
import KGFlow.utils as utils
import KGFlow.model as model
from KGFlow.metrics.ranking import *
from KGFlow.utils.embedding_utils import RandomInitEmbeddings
from KGFlow.data.kg import KG

from KGFlow.data import *

