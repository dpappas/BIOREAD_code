#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'Dimitris'

my_seed = 1989
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import os
import numpy as np
import cPickle as pickle
import torch.backends.cudnn as cudnn
import random
random.seed(my_seed)

cudnn.benchmark = True

embedding_dim   = 300
hidden_dim      = 192
learning_rate   = 0.0001 # 0.1 # 0.001
gpu_device      = 0

train_data_path = './data/bioread_subcorpus_batches_nonum_sorted/train/'
valid_data_path = './data/bioread_subcorpus_batches_nonum_sorted/valid/'
test_data_path  = './data/bioread_subcorpus_batches_nonum_sorted/test/'
od              = 'bioread_without_pn_asreader_rnn_from_scratch'
odir            = './bioread_pn/{}/'.format(od)
if not os.path.exists(odir):
    os.makedirs(odir)

import logging
logger = logging.getLogger(od)
hdlr = logging.FileHandler(odir+'model.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

resume_from     = None
start_epoch     = 0
vocab_size      = 650000

torch.manual_seed(my_seed)
print(torch.get_num_threads())
print(torch.cuda.is_available())
print(torch.cuda.device_count())

use_cuda = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(my_seed)

def print_params():
    print(40 * '=')
    print(model)
    print(40 * '=')
    total_params = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        total_params += v
    print(40 * '=')
    print(total_params)
    print(40 * '=')

def data_yielder(data_path, how_many):
    files = [ data_path+f for f in os.listdir(data_path) ]
    if(how_many is None):
        how_many = len(files)
    #shuffle(files)
    for file in files[:how_many]:
        try:
            d = pickle.load(open(file,'rb'))
            yield d['context'], d['quests'], d['cands'], d['targets']
        except:
            logger.error('yeilding error for file {}'.format(file))

def train_one_epoch(epoch):
    global sum_cost, sum_acc, m_batches
    gb = model.train()
    for b_context, b_quest, b_candidates, b_target in data_yielder(train_data_path, None):
        m_batches                   += 1
        optimizer.zero_grad()
        cost_, acc_, log_soft_res   = model( b_context, b_quest, b_candidates, b_target )
        cost_.backward()
        optimizer.step()
        sum_cost                    += cost_.data[0]
        sum_acc                     += acc_
        mean_cost                   = sum_cost / (m_batches * 1.0)
        mean_acc                    = sum_acc / (m_batches * 1.0)
        print(
            'train b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data[0], acc_, mean_cost, mean_acc
            )
        )
        logger.info(
            'train b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data[0], acc_, mean_cost, mean_acc
            )
        )
        #

def valid_one_epoch(epoch):
    gb = model.eval()
    sum_cost, sum_acc, m_batches = 0.0, 0.0, 0
    for b_context, b_quest, b_candidates, b_target in data_yielder(valid_data_path, None):
        m_batches                   += 1
        cost_, acc_, log_soft_res   = model( b_context, b_quest, b_candidates, b_target )
        sum_cost                    += cost_.data[0]
        sum_acc                     += acc_
        mean_cost                   = sum_cost / (m_batches * 1.0)
        mean_acc                    = sum_acc / (m_batches * 1.0)
        print(
            'valid b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data[0], acc_, mean_cost, mean_acc
            )
        )
        logger.info(
            'valid b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data[0], acc_, mean_cost, mean_acc
            )
        )
    return mean_cost

def test_one_epoch(epoch):
    gb = model.eval()
    sum_cost, sum_acc, m_batches = 0.0, 0.0, 0
    for b_context, b_quest, b_candidates, b_target in data_yielder(test_data_path, None):
        m_batches                   += 1
        cost_, acc_, log_soft_res   = model( b_context, b_quest, b_candidates, b_target )
        sum_cost                    += cost_.data[0]
        sum_acc                     += acc_
        mean_cost                   = sum_cost / (m_batches * 1.0)
        mean_acc                    = sum_acc / (m_batches * 1.0)
        print(
            'test b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data[0], acc_, mean_cost, mean_acc
            )
        )
        logger.info(
            'test b:{} e:{}. cost is: {} while accuracy is: {}. average_cost is: {} while average_accuracy is: {}'.format(
                m_batches, epoch, cost_.data[0], acc_, mean_cost, mean_acc
            )
        )

def dummy_test():
    b_context = np.random.randint(low=1, high=vocab_size - 1, size=(20, 5))
    b_context = np.concatenate([b_context, np.zeros(b_context.shape, dtype=np.int32)], axis=1)
    b_quest = np.random.randint(low=1, high=vocab_size - 1, size=(20, 4))
    b_quest = np.concatenate([b_quest, np.zeros(b_quest.shape, dtype=np.int32)], axis=1)
    b_candidates = np.unique(b_context[:, :4], axis=1)
    b_candidates = np.concatenate([b_candidates, np.zeros(b_candidates.shape, dtype=np.int32)], axis=1)
    b_target = np.array(b_candidates.shape[0] * [1])
    print(b_context.shape)
    print(b_quest.shape)
    print(b_candidates.shape)
    print(b_target.shape)
    model.train()
    for i in range(100):
        optimizer.zero_grad()
        cost_, acc_, log_soft_res = model(b_context, b_quest, b_candidates, b_target)
        print(cost_.data[0], acc_)
        cost_.backward()
        optimizer.step()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_model_from_checkpoint():
    global start_epoch
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint      = torch.load(resume_from)
        start_epoch     = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, return_sequences = True, gpu_device=None):
        super(BiRNN, self).__init__()
        self.gpu_device         = gpu_device
        self.return_sequences   = return_sequences
        self.hidden_size        = hidden_size
        self.onwards_gru        = nn.GRUCell(input_size, hidden_size)
        self.backwards_gru      = nn.GRUCell(input_size, hidden_size)
        if( use_cuda ):
            self.onwards_gru.cuda(self.gpu_device)
            self.backwards_gru.cuda(self.gpu_device)
        # if(use_cuda):
        #     self.onwards_gru.cuda()
        #     self.backwards_gru.cuda()
    def do_onwards(self, x, max_len):
        x = x.transpose(0, 1)
        output = []
        hx = autograd.Variable(torch.randn(x.size(1), self.hidden_size))
        if(use_cuda):
            hx = hx.cuda(self.gpu_device)
        # if(use_cuda):
        #     hx = hx.cuda()
        for i in range(max_len.data[0]):
            hx = self.onwards_gru(x[i], hx)
            hx = torch.tanh(hx)
            output.append(hx)
        output = torch.stack(output, dim=0).transpose(0, 1)
        return output
    def do_backwards(self, x, max_len):
        x = x.transpose(0, 1)
        output = []
        hx = autograd.Variable(torch.randn(x.size(1), self.hidden_size))
        if( use_cuda ):
            hx = hx.cuda(self.gpu_device)
        # if(use_cuda):
        #     hx = hx.cuda()
        for i in range(max_len.data[0]):
            hx = self.backwards_gru(x[x.size(0)-1-i], hx)
            hx = torch.tanh(hx)
            output.append(hx)
        output = [ output[i] for i in range(len(output)-1,-1,-1) ]
        output = torch.stack(output, dim=0).transpose(0, 1)
        return output
    def forward(self, x, lengths):
        variable_lengths = autograd.Variable(torch.LongTensor(lengths))
        # variable_lengths.requires_grad = False
        variable_lengths = variable_lengths.detach()
        if( use_cuda ):
            variable_lengths = variable_lengths.cuda(self.gpu_device)
        # if (use_cuda):
        #     variable_lengths = variable_lengths.cuda()
        max_len = variable_lengths.max()
        onwards_out = self.do_onwards(x, max_len)
        backwards_out = self.do_backwards(x, max_len)
        out = torch.cat( [onwards_out, backwards_out], dim=2)
        if(self.return_sequences):
            return out
        else:
            rnn_last_indices = (variable_lengths - 1).view(-1, 1).expand(x.size(0), out.size(2)).unsqueeze(1)
            last_state = out.gather(1, rnn_last_indices).squeeze(dim=1)
        return last_state

class ASReader_Modeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2):
        super(ASReader_Modeler, self).__init__()
        self.hidden_dim         = hidden_dim
        self.word_embeddings    = nn.Embedding(vocab_size, embedding_dim)
        self.context_gru        = BiRNN(embedding_dim, hidden_dim, True, gpu_device)
        self.question_gru       = BiRNN(embedding_dim, hidden_dim, False, gpu_device)
        self.softmax            = torch.nn.Softmax()
        self.dropout_f          = nn.Dropout(p=dropout_prob)
        self.dropout2D_f        = nn.Dropout2d(p=dropout_prob)
        if(use_cuda):
            self.context_gru.cuda(gpu_device)
            self.question_gru.cuda(gpu_device)
            self.dropout_f.cuda(gpu_device)
            self.dropout2D_f.cuda(gpu_device)
        # if(use_cuda):
        #     self.context_gru.cuda()
        #     self.question_gru.cuda()
        #     self.dropout_f.cuda()
        #     self.dropout2D_f.cuda()
    def get_candidates_for_inst(self, input, context, candidates):
        ret = None
        for cand in candidates:
            if(cand.data[0] == 0):
                pass
            else:
                mask        = torch.eq(context, cand)
                if (use_cuda):
                    mask        = mask.type(torch.cuda.FloatTensor)
                    mask        = mask.detach()
                    masked_soft = torch.mul(input.type(torch.cuda.FloatTensor), mask)
                else:
                    mask        = mask.type(torch.FloatTensor)
                    mask        = mask.detach()
                    masked_soft = torch.mul(input.type(torch.FloatTensor), mask)
                #
                mask        = torch.eq(candidates, cand)
                if (use_cuda):
                    mask        = mask.type(torch.cuda.FloatTensor)
                    mask = mask.detach()
                else:
                    mask        = mask.type(torch.FloatTensor)
                    mask = mask.detach()
                masked_cand = torch.mul(mask, torch.sum(masked_soft))
                #
                if(ret is None):
                    ret = masked_cand
                else:
                    ret = ret + masked_cand
        return ret
    def get_candidates(self, input, context, candidates):
        ret = []
        for i in range(input.size(0)):
            res_for_one_inst = self.get_candidates_for_inst(input[i], context[i], candidates[i])
            ret.append(res_for_one_inst)
        ret = torch.stack(ret,dim=0)
        return ret
    def mask_based_on(self, input, base, mask_value, replace_with):
        mask = torch.eq(torch.eq(base, mask_value), replace_with)
        if(use_cuda):
            mask = mask.type(torch.cuda.FloatTensor)
            mask = mask.detach()
        else:
            mask = mask.type(torch.FloatTensor)
            mask = mask.detach()
        ret = torch.mul( input, mask )
        return ret
    def masked_softmax(self, input, mask_value, replace_with):
        ret = torch.exp(input)
        ret = self.mask_based_on(ret, input, mask_value, replace_with)
        sum = torch.sum( ret, dim=1 )
        sum =  torch.stack(input.size(1) * [sum], dim=1)
        ret = torch.div(ret, sum)
        return ret
    def calculate_accuracy(self,soft_res,target):
        total       = (soft_res.size(0) * 1.0)
        soft_res    = np.argmax(soft_res.data.cpu().numpy(), axis=1)
        target      = target.data.cpu().numpy()
        wright_ones = len(np.where( soft_res == target)[0])
        acc         = wright_ones / total
        return  acc
    def forward(self, context, question, candidates, target):
        context                                 = autograd.Variable(torch.LongTensor(context))
        question                                = autograd.Variable(torch.LongTensor(question))
        candidates                              = autograd.Variable(torch.LongTensor(candidates))
        target                                  = autograd.Variable(torch.LongTensor(target))
        context.requires_grad                   = False
        question.requires_grad                  = False
        candidates.requires_grad                = False
        target.requires_grad                    = False
        #
        # if (use_cuda):
        #     context                             = context.cuda()
        #     question                            = question.cuda()
        #     candidates                          = candidates.cuda()
        #     target                              = target.cuda()
        #
        context_len                             = [torch.nonzero(item).size(0) for item in context.data]
        question_len                            = [torch.nonzero(item).size(0) for item in question.data]
        max_cands                               = torch.max(autograd.Variable(torch.LongTensor([torch.nonzero(item).size(0) for item in candidates.data])))
        # max_cands.requires_grad = False
        max_cands                               = max_cands.detach()
        max_c_len                               = torch.max(autograd.Variable(torch.LongTensor(context_len)))
        max_q_len                               = torch.max(autograd.Variable(torch.LongTensor(question_len)))
        #
        context                                 = context[:,:max_c_len.data[0]]
        question                                = question[:,:max_q_len.data[0]]
        candidates                              = candidates[:,:max_cands.data[0]]
        #
        if (use_cuda):
            context                             = context.cuda(gpu_device)
            question                            = question.cuda(gpu_device)
            candidates                          = candidates.cuda(gpu_device)
            target                              = target.cuda(gpu_device)
        #
        cont_embeds                             = self.word_embeddings(context)
        quest_embeds                            = self.word_embeddings(question)
        #
        cont_embeds                             = self.dropout_f(cont_embeds)
        quest_embeds                            = self.dropout_f(quest_embeds)
        # #
        # if (use_cuda):
        #     context                             = context.cuda(gpu_device)
        #     cont_embeds                         = cont_embeds.cuda(gpu_device)
        #     quest_embeds                        = quest_embeds.cuda(gpu_device)
        #     candidates                          = candidates.cuda(gpu_device)
        #     target                              = target.cuda(gpu_device)
        # #
        context_out                             = self.context_gru( cont_embeds, context_len)
        question_out                            = self.question_gru( quest_embeds, question_len )
        #
        question_out_stack                      = torch.stack( context_out.size(1) * [question_out], dim=1)
        element_mul                             = torch.mul( context_out, question_out_stack )
        #
        dot_p                                   = torch.sum( element_mul, dim=2 )
        dot_p                                   = self.mask_based_on( dot_p, context, 0, 0)
        dot_p_cands                             = self.get_candidates( dot_p, context, candidates)
        #
        log_soft_res                            = F.log_softmax(dot_p_cands)
        acc                                     = self.calculate_accuracy(log_soft_res, target)
        losss                                   = F.nll_loss(log_soft_res, target, weight=None, size_average=True)
        # print context.size()
        # print question.size()
        # print cont_embeds.size()
        # print quest_embeds.size()
        # print context_out.size()
        # print question_out.size()
        # print question_out_stack.size()
        # print element_mul.size()
        # print dot_p.size()
        # print dot_p_cands.size()
        # print log_soft_res.size()
        # print losss.size()
        # print 20 * '-'
        return losss, acc, log_soft_res

model = ASReader_Modeler(vocab_size, embedding_dim, hidden_dim)

if resume_from is not None:
    load_model_from_checkpoint()
else:
    print("=> no checkpoint found at '{}'".format(resume_from))

print_params()

if(use_cuda):
    model.cuda(gpu_device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

b_size              = 20
min_mean_valid_c    = 10000000
sum_cost, sum_acc, m_batches = 0., 0., 0
for epoch in range(start_epoch,20):
    train_one_epoch(epoch)
    mean_valid_c = valid_one_epoch(epoch)
    if(mean_valid_c < min_mean_valid_c):
        min_mean_valid_c = mean_valid_c
        test_one_epoch(epoch)
        state = {
            'epoch'         : epoch + 1,
            'state_dict'    : model.state_dict(),
            'best_cost'     : min_mean_valid_c,
            'optimizer'     : optimizer.state_dict(),
        }
        save_checkpoint(state, filename=odir+'best_checkpoint.pth.tar')

