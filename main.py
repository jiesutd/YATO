# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-03-01 01:20:54

from __future__ import print_function
import numpy as np
import os
import time
import sys
import random
import torch
import torch.optim as optim
from utils.metric import get_ner_fmeasure, get_sent_fmeasure
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

#os.environ['MKL_THREADING_LAYER'] = 'GNU'
#os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
try:
    import cPickle as pickle
except ImportError:
    import pickle

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data):
    """

    :param data:
    :return:
    """
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    """

    :param pred_variable:
    :param gold_variable:
    :param mask_variable:
    :param sentence_classification:
    :return:
    """
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.data.cpu().numpy()
    gold = gold_variable.data.cpu().numpy()
    mask = mask_variable.data.cpu().numpy()
    overlaped = (pred == gold)
    if sentence_classification:
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0]
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover,
                  sentence_classification=False):
    """

    :param pred_variable:
    :param gold_variable:
    :param mask_variable:
    :param label_alphabet:
    :param word_recover:
    :param sentence_classification:
    :return:
    """
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert (len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """

    :param pred_variable:
    :param mask_variable:
    :param label_alphabet:
    :param word_recover:
    :return:
    """
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
                         mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def evaluate(data, model, name, nbest=0):
    """

    :param data:
    :param model:
    :param name:
    :param nbest:
    :return:
    """
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    elif name == 'predict':
        instances = data.predict_Ids
    else:
        print("Error: wrong evaluate name," + str(name))
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    instance_num = len(instances)
    total_batch = instance_num // batch_size + 1
    for batch_id in tqdm(range(total_batch)):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > instance_num:
            end = instance_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_word_text, batch_label, mask = batchify_with_label(
            input_batch_list=instance, gpu=data.HP_gpu, device=data.device, if_train=True,
            sentence_classification=data.sentence_classification)
        if nbest > 1 and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, batch_word_text, None, mask,
                                                       nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            tag_seq = nbest_tag_seq[:, :, 0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                            batch_word_text, None, mask)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover,
                                               data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time

    if data.sentence_classification:
        acc, p, r, f = get_sent_fmeasure(gold_results, pred_results, list(set(data.sentence_tags)))
    else:
        acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest > 1 and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, float(acc), float(p), float(r), float(f), pred_results, pred_scores


def batchify_with_label(input_batch_list, gpu, device, if_train=True, sentence_classification=False):
    """

    :param input_batch_list:
    :param gpu:
    :param device:
    :param if_train:
    :param sentence_classification:
    :return:
    """
    if sentence_classification:
        return batchify_sentence_classification_with_label(input_batch_list, gpu, device, if_train)
    else:
        return batchify_sequence_labeling_with_label(input_batch_list, gpu, device, if_train)


def batchify_sequence_labeling_with_label(input_batch_list, gpu, device, if_train=True):
    """

    :param input_batch_list:
    :param gpu:
    :param device:
    :param if_train:
    :return:
    """
    """
        ## to incoperate the transformer, the input add the original word text
        input: list of words, chars and labels, various length. [[word_ids, feature_ids, char_ids, label_ids, words, features, chars, labels],[word_ids, feature_ids, char_ids, label_ids, words, features, chars, labels],...]

            word_Ids: word ids for one sentence. (batch_size, sent_len)
            feature_Ids: features ids for one sentence. (batch_size, sent_len, feature_num)
            char_Ids: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            label_Ids: label ids for one sentence. (batch_size, sent_len)
            words: word text for one sentence. (batch_size, sent_len)
            features: features text for one sentence. (batch_size, sent_len, feature_num)
            chars: char text for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label text for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
            batch_word_list: list of list, (batch_size, ) list of words for the batch, original order, not reordered, it will be reordered in transformer
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    batch_word_list = [sent[4] for sent in input_batch_list]
    feature_num = len(features[0][0])
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.to(device)
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].to(device)
        word_seq_lengths = word_seq_lengths.to(device)
        word_seq_recover = word_seq_recover.to(device)
        label_seq_tensor = label_seq_tensor.to(device)
        char_seq_tensor = char_seq_tensor.to(device)
        char_seq_recover = char_seq_recover.to(device)
        mask = mask.to(device)
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, batch_word_list, label_seq_tensor, mask


def batchify_sentence_classification_with_label(input_batch_list, gpu, device, if_train=True):
    """

    :param input_batch_list:
    :param gpu:
    :param device:
    :param if_train:
    :return:
    """
    """
        ## to incoperate the transformer, the input add the original word text
        input: list of words, chars and labels, various length. [[word_ids, feature_ids, char_ids, label_ids, words, features, chars, labels],[word_ids, feature_ids, char_ids, label_ids, words, features, chars, labels],...]
            word_ids: word ids for one sentence. (batch_size, sent_len)
            feature_ids: features ids for one sentence. (batch_size, feature_num), each sentence has one set of feature
            char_ids: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            label_ids: label ids for one sentence. (batch_size,), each sentence has one set of feature
            words: word text for one sentence. (batch_size, sent_len)
            ...
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size,), ... ] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, )
            mask: (batch_size, max_sent_len)
            batch_word_list: list of list, (batch_size, ) list of words for the batch, original order, not reordered, it will be reordered in transformer
    """

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    batch_word_list = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size,), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()
    label_seq_tensor = torch.LongTensor(labels)
    # exit(0)
    for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
    feature_seq_tensors = torch.LongTensor(np.swapaxes(np.asarray(features).astype(int), 0, 1))
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.to(device)
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].to(device)
        word_seq_lengths = word_seq_lengths.to(device)
        word_seq_recover = word_seq_recover.to(device)
        label_seq_tensor = label_seq_tensor.to(device)
        char_seq_tensor = char_seq_tensor.to(device)
        char_seq_recover = char_seq_recover.to(device)
        feature_seq_tensors = feature_seq_tensors.to(device)
        mask = mask.to(device)
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, batch_word_list, label_seq_tensor, mask


def train(data, log, metric):
    """

    :param data:
    :param log:
    :param metric:
    :return:
    """
    print("Training model...")
    logRecord = open(log, 'w', encoding='utf-8')
    save_data_name = data.dset_dir
    data.save(save_data_name)
    best_test = [{"acc": {"best test": 0, "best dev": 0, "epoch num": 0}},
                 {"f": {"best test": 0, "best dev": 0, "epoch num": 0}}]
    metric_seq = ["acc", 'f']
    batch_size = data.HP_batch_size
    instances = data.train_Ids
    instance_num = len(instances)
    total_step = instance_num // batch_size + 1
    total_steps = total_step * data.HP_iteration
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
    if data.scheduler.lower() == 'get_linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(total_steps * data.warmup_step_rate),
                                                    num_training_steps=total_steps)
    elif data.scheduler.lower() == 'get_cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(total_steps * data.warmup_step_rate),
                                                    num_training_steps=total_steps)

    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        instance_count = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_word_list, batch_label, mask = batchify_with_label(
                input_batch_list=instance, gpu=data.HP_gpu, device=data.device,
                sentence_classification=data.sentence_classification)
            instance_count += 1
            loss, tag_seq = model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                                 batch_charrecover, batch_word_list, batch_label, mask)

            if not data.sentence_classification:
                right, whole = predict_check(tag_seq, batch_label, mask)
                right_token += right
                whole_token += whole
            sample_loss += loss.item()
            total_loss += loss.item()
            model.zero_grad()
            if end % 500 == 0 and (not data.sentence_classification):
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                logRecord.writelines("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f \n" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0
            elif end % 500 == 0 and data.sentence_classification:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f;" % (
                    end, temp_cost, sample_loss))
                logRecord.writelines("     Instance: %s; Time: %.2fs; loss: %.4f; \n" % (
                    end, temp_cost, sample_loss))
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            if data.scheduler.lower() != 'None':
                scheduler.step()
            model.zero_grad()
        epoch_finish = time.time()
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = [acc, f]
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                dev_cost, speed, acc, p, r, f))
            logRecord.writelines("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f \n" % (
                dev_cost, speed, acc, p, r, f))
            sys.stdout.flush()
        else:
            current_score = [acc, f]
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f; f: %.4f" % (dev_cost, speed, acc, f))
            logRecord.writelines("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f; f: %.4f \n"
                                 % (dev_cost, speed, acc, f))
            sys.stdout.flush()
        speed, acc, p, r, f, _, _ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        test_current = [acc, f]
        for score, record, tscore, mtag in zip(current_score, best_test, test_current, metric_seq):
            trecord = record[mtag]
            if score > trecord["best dev"]:
                trecord["best test"] = tscore
                trecord["best dev"] = score
                trecord["epoch num"] = idx
                ex_model_name = data.model_dir + 'acc%.4f_p%.4f_r%.4f_f%.4f.pth' % (
                    acc, p, r, f)
                print("Save current best " + mtag + " model in file:" + str(ex_model_name))
                logRecord.writelines("Save current best " + mtag + " model in file:" + str(ex_model_name) + '\n')
                if not os.path.exists(ex_model_name):
                    torch.save(model.state_dict(), ex_model_name)
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
            logRecord.writelines("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f \n" % (
                test_cost, speed, acc, p, r, f))
            sys.stdout.flush()
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
            logRecord.writelines("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f \n" % (
                test_cost, speed, acc, p, r, f))
            sys.stdout.flush()
    if metric.lower() == 'a':
        best_test_record = best_test[0].get("acc")
        print('Best Test Accuracy: %s, Best Validation Accuracy: %s, Best Test Accuracy Epoch: %s' % (
            str(best_test_record["best test"]), str(best_test_record["best dev"]), str(best_test_record["epoch num"])))
        logRecord.writelines('Best Test Accuracy: %s, Best Validation Accuracy: %s, Best Test Accuracy Epoch: %s \n' % (
            str(best_test_record["best test"]), str(best_test_record["best dev"]), str(best_test_record["epoch num"])))
        sys.stdout.flush()
    elif metric.lower() == 'f':
        best_test_record = best_test[1].get("f")
        print('Best Test F1 Score: %s, Best Validation F1 Score: %s, Best Test F1 Score Epoch: %s' % (
            str(best_test_record["best test"]), str(best_test_record["best dev"]), str(best_test_record["epoch num"])))
        logRecord.writelines('Best Test F1 Score: %s, Best Validation F1 Score: %s, Best Test F1 Score Epoch: %s \n' % (
            str(best_test_record["best test"]), str(best_test_record["best dev"]), str(best_test_record["epoch num"])))
        sys.stdout.flush()


def load_model_decode(data, name):
    """

    :param data:
    :param name:
    :return:
    """
    print("Load Model from file: " + str(data.model_dir))
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)
    if data.HP_gpu == True or data.HP_gpu == 'True':
        model.load_state_dict(torch.load(data.load_model_dir))
    else:
        model.load_state_dict(torch.load(data.load_model_dir, map_location='cpu'))

    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores= evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return speed, acc, p, r, f, pred_results, pred_scores


def run_ncrfpp(config, log='test.log', metric='F', wordemb=None, charemb=None, savemodel=None,
               savedset=None, train_path="data/conll03/train.bmes", dev_path="data/conll03/dev.bmes",
               test_path="data/conll03/test.bmes", seg='True', loadmodel=None):
    data = Data()
    if config == 'None':
        data.train_dir = train_path
        data.dev_dir = dev_path
        data.test_dir = test_path
        data.model_dir = savemodel
        data.load_model_dir = loadmodel
        data.dset_dir = savedset
        data.word_emb_dir = wordemb
        data.char_emb_dir = charemb
        if seg.lower() == 'true':
            data.seg = True
        else:
            data.seg = False
    status = data.status.lower()
    print("Seed num:%s", str(seed_num))
    if status == 'train':
        print("MODEL: train")
        data.read_config(config)
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        data.summary()
        train(data, log, metric)

    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(config)
        data.generate_instance('raw')
        print("nbest: %s" % (data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest > 0 and not data.sentence_classification:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    elif status == 'prob':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(config)
        print(str(data.raw_dir))
        data.generate_instance('raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")


if __name__ == '__main__':
    run_ncrfpp(config='../test.config')
