# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:21:52 2024

@author: User
"""

import os
import json
import pandas as pd

from tqdm import tqdm

from transformers import BertModel, BertConfig
import torch
from torch.autograd import grad

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
from config.config import args

def find_topic_words(sent, label):
    
    topic_str = ''
    m=0
    topics=[]
    
    while m<len(sent):
        
        topic = label[m]
        if topic==1:
            topic_str+=sent[m]
            
            for n in range(1,5):
                topic = label[m+n]
                if topic==2:
                    topic_str = topic_str + ' ' + sent[m+n]
                if topic==1 or topic==0:
                    m=m+n
                    topics.append(topic_str)
                    topic_str = ''
                    break
        else:
            m+=1
    topics = (', '.join(topics))
    return topics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = data_utils.AeProcessor()
label_list = processor.get_labels()
tokenizer = ABSATokenizer.from_pretrained(args['pretrained_model_path'])
eval_examples = processor.get_test_examples(args['data_dir'])
eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args['max_seq_length'], tokenizer, "ae")

# logger.info("***** Running evaluation *****")
# logger.info("  Num examples = %d", len(eval_examples))
# logger.info("  Batch size = %d", args.eval_batch_size)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
bs = 4
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=bs)

model = torch.load(os.path.join(args['output_dir'], "model.pt"), map_location=torch.device('cpu'))
model.to(device)
model.eval()

full_logits=[]
full_label_ids=[]
feedback_list = []
topic_list = []

for step, batch in enumerate(eval_dataloader):
    batch = tuple(t.to(device) for t in batch)
    input_ids, segment_ids, input_mask, label_ids = batch
    
    
    with torch.no_grad():
        logits = model(input_ids, input_mask)
        
    print(logits.shape)
    
    logits = torch.argmax(logits, dim=2)

    # Unsqueeze to add a new dimension along the third axis
    logits = logits.unsqueeze(2)

    logits = logits.detach().cpu().numpy()
    
    for j in range(logits.shape[0]):
        # print(step*bs+j)
            
        feedback = eval_examples[step*bs+j].text_a
        topic_label_id = list(logits[j,1:len(feedback)+2,0])
        
        try:
            topic = find_topic_words(feedback, topic_label_id)
        except:
            topic = 'Topic finding logic fails'
        
        feedback_list.append(feedback)
        topic_list.append(topic)
        
df = pd.DataFrame({'feedbacks':feedback_list, 'topics':topic_list})
df.to_csv('topic.csv', index=False)
