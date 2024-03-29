# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-01 21:11:50
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-29 14:27:55

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
import numpy as np
from .classificationhead import ClassificationHead

class SentClassifier(nn.Module):
    def __init__(self, data):
        super(SentClassifier, self).__init__()
        if not data.silence:
            print("build sentence classification network...")
            print("use_char: ", data.use_char)
            if data.use_char:
                print("char feature extractor: ", data.char_feature_extractor)
            print("word feature extractor: ", data.word_feature_extractor)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.label_size = data.label_alphabet_size
        self.classifier = data.classification_head
        self.word_hidden = WordSequence(data).to(data.device)
        if self.classifier:
            self.classifier = ClassificationHead(hidden_size=self.word_hidden.output_hidden_dim, \
                                                 activation_function=data.classification_activation,
                                             num_labels=data.label_alphabet_size, classifier_dropout=data.classifier_dropout,
                                             dropout_prob=data.HP_dropout).to(data.device)
    def calculate_loss(self, *input):
        ## input = word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_word_text, batch_label, mask
        outs, _ = self.word_hidden.sentence_representation(*input)
        
        word_inputs = input[0]
        batch_label = input[7]
        batch_size = word_inputs.size(0)
        if self.classifier:
            outs = self.classifier(outs)
        outs = outs.view(batch_size, -1)
        
        #loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        #total_loss = loss_fct(outs.view(-1, self.label_size), batch_label.view(-1))
        
        total_loss = F.cross_entropy(outs, batch_label.view(batch_size), ignore_index=0)
        
        _, tag_seq  = torch.max(outs, 1)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, *input):
        ## input = word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_word_text, mask,...
        word_inputs = input[0]
        outs, _ = self.word_hidden.sentence_representation(*input)
        batch_size = word_inputs.size(0)
        if self.classifier:
            outs = self.classifier(outs)
        outs = outs.view(batch_size, -1)
        _, tag_seq  = torch.max(outs, 1)
        return tag_seq

    def get_target_probability(self, *input):
        # input = word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_word_text, mask
        word_inputs = input[0]
        outs, weights = self.word_hidden.sentence_representation(*input)
        batch_size = word_inputs.size(0)
        outs = outs.view(batch_size, -1)
        _, tag_seq  = torch.max(outs, 1)
        outs = outs[:,1:]
        sf = nn.Softmax(1)
        prob_outs = sf(outs)
        if self.gpu:
            prob_outs = prob_outs.cpu()
            if type(weights) != type(None):
                weights = weights.cpu()
    
        if type(weights) != type(None):
            weight = weights.detach().numpy()

        probs = np.insert(prob_outs.detach().numpy(), 0, 0, axis=1)
        
        return probs, weights


