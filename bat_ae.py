# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 15:43:19 2023

@author: User
"""

from transformers import BertPreTrainedModel, BertModel
import torch
from torch.autograd import grad

class BertForABSA(BertModel):
    def __init__(self, config, num_labels=3, dropout=None, epsilon=None):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.epsilon = epsilon
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, bert_emb = self.bert_forward(input_ids, 
                                                token_type_ids, 
                                                attention_mask, 
                                                output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            _loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if sequence_output.requires_grad: #if training mode
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