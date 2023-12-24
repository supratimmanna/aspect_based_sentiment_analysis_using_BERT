# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:48:20 2023

@author: User
"""
import os
import json

from transformers import BertPreTrainedModel, BertModel, BertConfig
import torch
from torch.autograd import grad

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer

# from config.config import args

class BertForTopicExtraction(BertModel):
    def __init__(self, model_path, config, num_labels=3, dropout=None, epsilon=None):
        super(BertForTopicExtraction, self).__init__(config)
        
        self.model = BertModel.from_pretrained(model_path)
        self.num_labels = num_labels
        self.epsilon = epsilon
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, adv_attack=False):
        if adv_attack:
            sequence_output, bert_emb = self.bert_forward(input_ids, 
                                                    token_type_ids, 
                                                    attention_mask, 
                                                    output_all_encoded_layers=False)
        
        else:
            sequence_output = self.model(input_ids, attention_mask)[0]
        
            
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            _loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if sequence_output.requires_grad and adv_attack: #if training mode
                perturbed_sentence = self.adv_attack(bert_emb, _loss, self.epsilon)
                adv_loss = self.adversarial_loss(perturbed_sentence, attention_mask, labels)
                return _loss, adv_loss
            return _loss
        else:
            return logits

    def adv_attack(self, emb, loss, epsilon):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_sentence = emb + epsilon * (loss_grad/(loss_grad_norm.reshape(-1,1,1)))
        return perturbed_sentence

    def adversarial_loss(self, perturbed, attention_mask, labels):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.encoder(perturbed, extended_attention_mask, 
                                            output_all_encoded_layers=False)
        encoded_layers_last = encoded_layers[-1]
        encoded_layers_last = self.dropout(encoded_layers_last)
        logits = self.classifier(encoded_layers_last)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        adv_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss

    def bert_forward(self, input_ids, token_type_ids=None, 
                        attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, 
                                        extended_attention_mask, 
                                        output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        
        return sequence_output, embedding_output
    
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
            valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)    
    
            best_valid_loss=float('inf')
            valid_losses=[]
            
        # model_config = BertConfig.from_pretrained(args['pretrained_model_path'])
        # model = BertForTopicExtraction(args['pretrained_model_path'], model_config, num_labels = len(label_list), 
        #                                                 dropout=args['dropout'], epsilon=args['epsilon'])
        # self.model = self.bert_model
        self.model.to(device)
        
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=args['lr'])
        
        global_step = 0
        self.model.train()
        
        for _ in range(args['num_train_epochs']):
            for step, batch in enumerate(train_dataloader):
                
                if step==0:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, segment_ids, input_mask, label_ids = batch
                    print(input_ids.shape)
                    print(segment_ids.shape)
                    print(input_mask.shape)
                    print(label_ids.shape)
        
        
                    # _loss, adv_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                    _loss = self.model(input_ids, input_mask)
        #             loss = _loss #+ adv_loss
        #             loss.backward()
                    
        #             # lr_this_step = args['lr'] * warmup_linear(global_step/t_total, args.warmup_proportion)
        #             # for param_group in optimizer.param_groups:
        #             #     param_group['lr'] = lr_this_step
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             global_step += 1
                
            #>>>> perform validation at the end of each epoch .
        #     if args['do_valid']:
                
        #         self.model.eval()
        #         with torch.no_grad():
        #             losses=[]
        #             valid_size=0
        #             for step, batch in enumerate(valid_dataloader):
        #                 batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
        #                 input_ids, segment_ids, input_mask, label_ids = batch
        #                 loss = self.model(input_ids, segment_ids, input_mask, label_ids)
        #                 losses.append(loss.data.item()*input_ids.size(0) )
        #                 valid_size+=input_ids.size(0)
        #             valid_loss=sum(losses)/valid_size
        #             logger.info("validation loss: %f", valid_loss)
        #             valid_losses.append(valid_loss)
        #         # if valid_loss<best_valid_loss:
        #         #     torch.save(model, os.path.join(args['output_dir'], "model.pt") )
        #         #     best_valid_loss=valid_loss
        #         self.model.train()
        
        # if args['do_valid']:
        #     if not os.path.exists(args['output_dir']):
        #         os.mkdir(args['output_dir'])
                
        #     with open(os.path.join(args['output_dir'], "valid.json"), "w") as fw:
                # json.dump({"valid_losses": valid_losses}, fw)
        # else:
        #     torch.save(model, os.path.join(args.output_dir, "model.pt") )