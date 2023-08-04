'''
 # @ Author: Jie Yang
 # @ Create Time: 2020-03-28 00:03:18
 # @ Last Modified by: Jie Yang  Contact: jieynlp@gmail.com
 # @ Last Modified time: 2020-04-04 02:47:54
 '''

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .custommodel import CustomModel


class NCRFTransformers(nn.Module):
    def __init__(self, model_name, device, customfig='none', customTokenizer='none', customModel='none',
                 fix_embeddings=False):

        super(NCRFTransformers, self).__init__()
        print("Loading transformer... model:", model_name)
        self.device = device
        if (customTokenizer.lower() == 'none' or customTokenizer is None) and (
                customModel.lower() == 'none' or customModel is None):
            self.model_class, self.tokenizer_class, self.pretrained_weights = \
                AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(
                    model_name, use_fast=False), model_name
            self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
            self.model = self.model_class.from_pretrained(self.pretrained_weights).to(self.device)
        else:
            print('!!' * 10)
            print('USE CUSTOM MODEL CONFIG AND TOKENIZER')
            print('!!' * 10)
            self.pretrained_weights = model_name
            self.customfig = customfig
            self.CModel = CustomModel(customConfig=customfig, customTokenizer=customTokenizer, customModel=customModel)
            token_fun = getattr(self.CModel, customTokenizer.lower())
            self.tokenizer_class = token_fun(self.pretrained_weights)
            model_fun = getattr(self.CModel, customModel.lower())
            self.model_class = model_fun()
            self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
            self.model = self.model_class.from_pretrained(self.pretrained_weights).to(self.device)
        self.hidden_dim = self.model.config.hidden_size
        self.max_length = self.model.config.max_position_embeddings
        if fix_embeddings:
            for name, param in self.model.named_parameters():
                if name.startswith('embeddings'):
                    param.requires_grad = False
        print(" " + "++" * 20)
        print(self.model.config)
        print(" " + "++" * 20)
    def extract_features(self, input_batch_list, device):
        ## extract word list and calculate max_word seq len, get rank order (word_perm_idx) to fit other network settings (e.g. LSTM)
        batch_size = len(input_batch_list)
        words = [sent for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        max_word_seq_len = word_seq_lengths.max().item()
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        ## tokenize the input words, calculate the max token seq len. add the subword index vector. Currently only use non-batch method to recover subword(token)->word
        batch_tokens = []
        batch_token_ids = []
        subword_word_indicator = torch.zeros((batch_size, max_word_seq_len), dtype=torch.int64)
        for idx in range(batch_size):
            one_sent_token = []
            one_subword_word_indicator = []
            for word in input_batch_list[idx]:
                if 'http' in word or 'https' in word:
                    word = 'url'
                word_tokens = self.tokenizer.tokenize(word)
                one_subword_word_indicator.append(len(one_sent_token) + 1)
                one_sent_token += word_tokens
            ## add [cls] and [sep] tokens, only for classific BERT, GPT have different type
            one_sent_token = ['[CLS]'] + one_sent_token + ['[SEP]']
            one_sent_token_id = self.tokenizer.convert_tokens_to_ids(one_sent_token)
            batch_tokens.append(one_sent_token)
            batch_token_ids.append(one_sent_token_id)
            subword_word_indicator[idx, :len(one_subword_word_indicator)] = torch.LongTensor(one_subword_word_indicator)
        ## calculate the max token number
        token_seq_lengths = torch.LongTensor(list(map(len, batch_tokens)))
        max_token_seq_len = token_seq_lengths.max().item()

        ## padding token ids and generate tensor
        batch_token_ids_padded = []
        for the_ids in batch_token_ids:
            batch_token_ids_padded.append(the_ids + [0] * (max_token_seq_len - len(the_ids)))
            ## need token mask? maybe not necessary
            ##batch_token_mask.append([1]*len(the_ids) + [0]*(max_token_seq_len-len(the_ids)))

        ## reorder batch instances to fit other network settings
        batch_token_ids_padded_tensor = torch.tensor(batch_token_ids_padded)[word_perm_idx].to(self.device)
        subword_word_indicator = subword_word_indicator[word_perm_idx].to(self.device)  ## subword-> word mapping
        last_hidden_states = self.model(batch_token_ids_padded_tensor)[0]  # Models outputs are now tuples
        ## recover the batch token to word level representation. Four ways of merging subwords to words,
        # i.e. max-pooling, min-pooling, average-pooling, first-subword-selection. Currently only use non-batch method to recover.
        # Current supports first-subword only
        batch_word_mask_tensor_list = []

        for idx in range(batch_size):
            one_sentence_vector = torch.index_select(last_hidden_states[idx], 0, subword_word_indicator[idx]).unsqueeze(
                0)
            batch_word_mask_tensor_list.append(one_sentence_vector)
        batch_word_mask_tensor = torch.cat(batch_word_mask_tensor_list, 0)

        ## extract sequence representation, current only use the first token (i.e. [CLS]) as the sequence representation
        sequence_tensor = last_hidden_states[:, 0, :]

        return batch_word_mask_tensor.to(device), sequence_tensor.to(device)
