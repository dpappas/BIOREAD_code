
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import cPickle as pickle
import numpy as np
import random
random.seed(1989)
import os
import sys
import time
import shutil

cudnn.benchmark = True

torch.manual_seed(1989)
print torch.get_num_threads()
print torch.cuda.is_available()
print torch.cuda.device_count()

use_cuda = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(1989)

od              = 'bioread_with_pn_aoareader_rnn_from_scratch'
odir            = './bioread_pn_output/{}/'.format(od)
if not os.path.exists(odir):
    os.makedirs(odir)

start_epoch     = 0
resume_from     = None
vocab_size      = 650000
embedding_dim   = 300
hidden_dim      = 192
learning_rate   = 0.0001
gpu_device      = 0
train_data_path = './data/bioread_subcorpus_batches_nonum_sorted/train/'
valid_data_path = './data/bioread_subcorpus_batches_nonum_sorted/valid/'
test_data_path  = './data/bioread_subcorpus_batches_nonum_sorted/test/'
train_how_many  = 10000

import logging
params      = [ od, odir+'model.log', ]
logger      = logging.getLogger(params[0])
hdlr        = logging.FileHandler(params[1])
formatter   = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

def data_yielder(data_path, how_many):
    files = [ data_path+f for f in os.listdir(data_path) ]
    random.shuffle(files)
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
    for b_context, b_quest, b_candidates, b_target in data_yielder(train_data_path, how_many=train_how_many):
        m_batches                   += 1
        optimizer.zero_grad()
        cost_, acc_, log_soft_res, soft_res   = model( b_context, b_quest, b_candidates, b_target )
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

def valid_one_epoch(epoch):
    gb = model.eval()
    sum_cost, sum_acc, m_batches = 0.0, 0.0, 0
    for b_context, b_quest, b_candidates, b_target in data_yielder(valid_data_path, None):
        m_batches                   += 1
        cost_, acc_, log_soft_res, soft_res   = model( b_context, b_quest, b_candidates, b_target )
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
        cost_, acc_, log_soft_res, soft_res   = model( b_context, b_quest, b_candidates, b_target )
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
    print b_context.shape
    print b_quest.shape
    print b_candidates.shape
    print b_target.shape
    model.train()
    for i in range(100):
        optimizer.zero_grad()
        cost_, acc_, log_soft_res, soft_res= model(b_context, b_quest, b_candidates, b_target)
        print cost_.data[0], acc_
        cost_.backward()
        optimizer.step()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_model_from_checkpoint():
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        # checkpoint      = torch.load(resume_from)
        # checkpoint      = torch.load(resume_from, map_location=lambda storage, loc: storage)
        checkpoint      = torch.load(resume_from)
        start_epoch     = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

def print_params():
    print 40 * '='
    print model
    print 40 * '='
    total_params = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        total_params += v
    print 40 * '='
    print total_params
    print 40 * '='

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, return_sequences = True):
        super(BiRNN, self).__init__()
        self.return_sequences   = return_sequences
        self.hidden_size        = hidden_size
        self.onwards_gru        = nn.GRUCell(input_size, hidden_size)
        self.backwards_gru      = nn.GRUCell(input_size, hidden_size)
        if(use_cuda):
            self.onwards_gru.cuda(gpu_device)
            self.backwards_gru.cuda(gpu_device)
    def do_onwards(self, x, max_len):
        x = x.transpose(0, 1)
        output = []
        hx = autograd.Variable(torch.randn(x.size(1), self.hidden_size))
        if(use_cuda):
            hx = hx.cuda(gpu_device)
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
        if(use_cuda):
            hx = hx.cuda(gpu_device)
        for i in range(max_len.data[0]):
            hx = self.backwards_gru(x[x.size(0)-1-i], hx)
            hx = torch.tanh(hx)
            output.append(hx)
        output = [ output[i] for i in range(len(output)-1,-1,-1) ]
        output = torch.stack(output, dim=0).transpose(0, 1)
        return output
    def forward(self, x, lengths):
        variable_lengths = autograd.Variable(torch.LongTensor(lengths))
        if(use_cuda):
            variable_lengths = variable_lengths.cuda(gpu_device)
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

class AOAReader_Modeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.2):
        super(AOAReader_Modeler, self).__init__()
        self.hidden_dim         = hidden_dim
        self.word_embeddings    = nn.Embedding(vocab_size, embedding_dim)
        self.context_gru        = BiRNN(embedding_dim, hidden_dim, True)
        self.question_gru       = BiRNN(embedding_dim, hidden_dim, True)
        self.softmax            = torch.nn.Softmax()
        self.dropout_f          = nn.Dropout(p=dropout_prob)
        if(use_cuda):
            self.dropout_f.cuda(gpu_device)
            self.word_embeddings.cpu()
            self.question_gru.cuda(gpu_device)
            self.context_gru.cuda(gpu_device)
    def get_candidates_for_inst(self, input, context, candidates):
        ret = None
        for cand in candidates:
            if(cand.data[0] == 0):
                pass
            else:
                mask        = torch.eq(context, cand)
                if (use_cuda):
                    mask        = mask.type(torch.cuda.FloatTensor)
                    masked_soft = torch.mul(input.type(torch.cuda.FloatTensor), mask)
                else:
                    mask        = mask.type(torch.FloatTensor)
                    masked_soft = torch.mul(input.type(torch.FloatTensor), mask)
                #
                mask        = torch.eq(candidates, cand)
                if (use_cuda):
                    mask        = mask.type(torch.cuda.FloatTensor)
                else:
                    mask        = mask.type(torch.FloatTensor)
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
        mask = mask.type(torch.FloatTensor)
        if(use_cuda):
            mask = mask.cuda()
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
    def softmax_across_rows(self, M):
        ret = []
        for row in M :
            ret.append(self.softmax(torch.unsqueeze(row,0)))
        ret = torch.stack(ret, dim=0).squeeze(1)
        # print ret.size()
        return ret
    def get_pairwise_score(self, con_gru_out, quest_gru_out):
        rows_att = []
        cols_att = []
        for inst_counter in range(con_gru_out.size(0)):
            M = torch.mm( con_gru_out[inst_counter], quest_gru_out[inst_counter].transpose(1,0) )
            rows_att.append(self.softmax_across_rows(M))
            cols_att.append(self.softmax_across_rows(M.transpose(1,0)).transpose(0,1))
        #
        rows_att = torch.stack(rows_att)
        av = rows_att.sum(1)/ ( rows_att.size(1) * 1.0)
        #
        cols_att = torch.stack(cols_att)
        #
        o = []
        for inst_counter in range(cols_att.size(0)):
            o.append(torch.mm(cols_att[inst_counter], torch.unsqueeze(av[inst_counter], 1)))
        o = torch.stack(o)
        o = o.squeeze(-1)
        # print o
        return o
    def forward(self, context, question, candidates, target):
        context                                 = autograd.Variable(torch.LongTensor(context), requires_grad=False)
        question                                = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        candidates                              = autograd.Variable(torch.LongTensor(candidates), requires_grad=False)
        target                                  = autograd.Variable(torch.LongTensor(target), requires_grad=False)
        if(use_cuda):
            context                             = context.cuda(gpu_device)
            question                            = question.cuda(gpu_device)
            candidates                          = candidates.cuda(gpu_device)
            target                              = target.cuda(gpu_device)
        #
        context_len                             = [torch.nonzero(item).size(0) for item in context.data]
        question_len                            = [torch.nonzero(item).size(0) for item in question.data]
        #
        max_cands                               = torch.max(autograd.Variable(torch.LongTensor([torch.nonzero(item).size(0) for item in candidates.data])))
        max_c_len                               = torch.max(autograd.Variable(torch.LongTensor(context_len)))
        max_q_len                               = torch.max(autograd.Variable(torch.LongTensor(question_len)))
        #
        context                                 = context[:,    :max_c_len.data[0]]
        question                                = question[:,   :max_q_len.data[0]]
        candidates                              = candidates[:, :max_cands.data[0]]
        #
        cont_embeds                             = self.word_embeddings(context)
        quest_embeds                            = self.word_embeddings(question)
        #
        cont_embeds                             = self.dropout_f(cont_embeds)
        quest_embeds                            = self.dropout_f(quest_embeds)        #
        #
        context_out                             = self.context_gru( cont_embeds, context_len)
        question_out                            = self.question_gru( quest_embeds, question_len )
        #
        pws                                     = self.get_pairwise_score(context_out, question_out)
        pws_cands                               = self.get_candidates( pws, context, candidates)
        # print pws_cands.size()
        #
        log_soft_res                            = F.log_softmax(pws_cands)
        soft_res                                = F.softmax (pws_cands)
        acc                                     = self.calculate_accuracy(log_soft_res, target)
        losss                                   = F.nll_loss(log_soft_res, target, weight=None, size_average=True)
        return losss, acc, log_soft_res, soft_res

model           = AOAReader_Modeler(vocab_size, embedding_dim, hidden_dim)
optimizer       = optim.Adam(model.parameters(), lr=learning_rate)

if(use_cuda):
    model.cuda(gpu_device)

if resume_from is not None:
    load_model_from_checkpoint()
else:
    print("=> no checkpoint found at '{}'".format(resume_from))

print_params()

if(use_cuda):
    model.cuda(gpu_device)

min_mean_valid_c = 9000000
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



