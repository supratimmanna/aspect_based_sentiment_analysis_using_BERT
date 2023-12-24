# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:42:11 2023

@author: User
"""

import os
import logging
import argparse
import random
import json
from math import ceil
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import BertAdam
from transformers import BertTokenizer
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertPreTrainedModel, BertModel, BertConfig
# from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
# import modelconfig

# from bert_topic_extraction_model import BertForTopicExtraction

from topic_extraction import Bert_Model, BertForTopicExtraction
from config.config import args

        

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = data_utils.AeProcessor()
label_list = processor.get_labels()

model_config = BertConfig.from_pretrained(args['pretrained_model_path'])

model = BertForTopicExtraction(args['pretrained_model_path'], model_config, 
                                num_labels = len(label_list), dropout=args['dropout'])



model.fine_tuning(args, logger)

