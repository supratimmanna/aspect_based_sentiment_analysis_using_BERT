# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:48:20 2023

@author: User
"""
import os
import json

from tqdm import tqdm

from transformers import BertModel, BertConfig
import torch
from torch.autograd import grad

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer

# from config.config import args

class Bert_Model(torch.nn.Module):
    def __init__(self, model_path, config, num_labels=3, dropout=0.1):
        super(Bert_Model, self).__init__()
        
        self.bert_model = BertModel.from_pretrained(model_path)
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, labels=None):
        
        sequence_output = self.bert_model(input_ids, attention_mask)[0]
        
            
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            _loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return _loss
        else:
            return logits


class BertForTopicExtraction():
    def __init__(self, model_path, config, num_labels=3, dropout=0.1):
        
        self.model = Bert_Model(model_path, config, num_labels=3, dropout=dropout)
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    
    def fine_tuning(self, args, logger):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        processor = data_utils.AeProcessor()
        label_list = processor.get_labels()
        
        ## Create the tokenizer
        tokenizer = ABSATokenizer.from_pretrained(args['pretrained_model_path'])
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_examples = processor.get_train_examples(args['data_dir'])
        
        num_train_steps = int(len(train_examples) / args['train_batch_size']) * args['num_train_epochs']
        
        train_features = data_utils.convert_examples_to_features(
                train_examples, label_list, args['max_seq_length'], tokenizer, "ae")
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args['train_batch_size'])
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])
        
        
        #>>>>> validation
        
        if args['do_valid']:
            valid_examples = processor.get_dev_examples(args['data_dir'])
            
            valid_features=data_utils.convert_examples_to_features(
                valid_examples, label_list, args['max_seq_length'], tokenizer, "ae")
            
            valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
            valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
            valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
            valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
            
            valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids)
    
            logger.info("***** Running validations *****")
            logger.info("  Num orig examples = %d", len(valid_examples))
            logger.info("  Num split examples = %d", len(valid_features))
            logger.info("  Batch size = %d", args['train_batch_size'])
    
            valid_sampler = SequentialSampler(valid_data)
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args['train_batch_size'])    
    
            best_valid_loss=float('inf')
            valid_losses=[]
            
        self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args['lr'])
        
        global_step = 0
        self.model.train()
        
        all_train_loss = []
        all_val_loss = []
        
        
        for iteration in tqdm(range(args['num_train_epochs'])):
            train_losses = []
            train_size = 0
            for step, batch in enumerate(train_dataloader):
                
                print('Epochs: {}, Step: {}'.format(iteration, step))
                
                # if step==0:
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_mask, label_ids = batch
                
    
                # _loss, adv_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                _loss = self.model(input_ids, input_mask, label_ids)
                loss = _loss #+ adv_loss
                loss.backward()
                
                # lr_this_step = args['lr'] * warmup_linear(global_step/t_total, args.warmup_proportion)
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                train_losses.append(loss.data.item()*input_ids.size(0) )
                train_size+=input_ids.size(0)
                
                
                logger.info("Training loss : %f", loss.data.item())
                    
            train_loss = sum(train_losses)/train_size
            all_train_loss.append(train_loss)
                
            #>>>> perform validation at the end of each epoch .
            if args['do_valid']:
                
                self.model.eval()
                with torch.no_grad():
                    losses=[]
                    valid_size=0
                    for step, batch in enumerate(valid_dataloader):
                        # if step==0:
                        batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                        input_ids, segment_ids, input_mask, label_ids = batch
                        loss = self.model(input_ids, input_mask, label_ids)
                        losses.append(loss.data.item()*input_ids.size(0))
                        valid_size+=input_ids.size(0)
                    valid_loss=sum(losses)/valid_size
                    logger.info("validation loss: %f", valid_loss)
                    valid_losses.append(valid_loss)
                # if valid_loss<best_valid_loss:
                #     torch.save(model, os.path.join(args['output_dir'], "model.pt") )
                #     best_valid_loss=valid_loss
                all_val_loss.append(valid_loss)
                self.model.train()
        
        loss_info = {'training_loss': all_train_loss, 'validation_loss':all_val_loss}
            
        if args['do_valid']:
            if not os.path.exists(args['output_dir']):
                os.mkdir(args['output_dir'])
                
            with open(os.path.join(args['output_dir'], "valid.json"), "w") as fw:
                json.dump({"valid_losses": valid_losses}, fw)
        else:
            if not os.path.exists(args['output_dir']):
                os.mkdir(args['output_dir'])
            torch.save(self.model, os.path.join(args['output_dir'], "model.pt") )
            
        return loss_info
            