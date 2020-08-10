import sys
import os
import datetime
from utils import augment_triplet, evaluate

dataset = 'data/FB15k'
path = './record'

iterations = 2

kge_model = 'TransE'
kge_batch = 1024
kge_neg = 256
kge_dim = 100
kge_gamma = 24
kge_alpha = 1
kge_lr = 0.001
kge_iters = 10000
kge_tbatch = 16
kge_reg = 0.0
kge_topk = 100

if kge_model == 'RotatE':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 24.0, 1.0, 0.0001, 150000, 16
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 9.0, 1.0, 0.00005, 100000, 16
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 12.0, 0.5, 0.0001, 80000, 8
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 6.0, 0.5, 0.00005, 80000, 8

if kge_model == 'TransE':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 24.0, 1.0, 0.0001, 150000, 16
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 9.0, 1.0, 0.00005, 100000, 16
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 12.0, 0.5, 0.0001, 80000, 8
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 6.0, 0.5, 0.00005, 80000, 8

if kge_model == 'DistMult':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 2000, 500.0, 1.0, 0.001, 150000, 16, 0.000002
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 2000, 200.0, 1.0, 0.001, 100000, 16, 0.00001
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 200.0, 1.0, 0.001, 80000, 8, 0.00001
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 200.0, 1.0, 0.002, 80000, 8, 0.000005

if kge_model == 'ComplEx':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 500.0, 1.0, 0.001, 150000, 16, 0.000002
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 200.0, 1.0, 0.001, 100000, 16, 0.00001
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 500, 200.0, 1.0, 0.001, 80000, 8, 0.00001
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 500, 200.0, 1.0, 0.002, 80000, 8, 0.000005

if dataset.split('/')[-1] == 'FB15k':
    mln_threshold_of_rule = 0.1
    mln_threshold_of_triplet = 0.7
    weight = 0.5
if dataset.split('/')[-1] == 'FB15k-237':
    mln_threshold_of_rule = 0.6
    mln_threshold_of_triplet = 0.7
    weight = 0.5
if dataset.split('/')[-1] == 'wn18':
    mln_threshold_of_rule = 0.1
    mln_threshold_of_triplet = 0.5
    weight = 100
if dataset.split('/')[-1] == 'wn18rr':
    mln_threshold_of_rule = 0.1
    mln_threshold_of_triplet = 0.5
    weight = 100

mln_iters = 1000
mln_lr = 0.0001
mln_threads = 8

# ------------------------------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def cmd_kge(workspace_path, model):
    if model == 'RotatE':
        return 'bash ./kge/kge.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {} -de'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk)
    if model == 'TransE':
        return 'bash ./kge/kge.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {}'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk)
    if model == 'DistMult':
        return 'bash ./kge/kge.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {} -r {}'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk, kge_reg)
    if model == 'ComplEx':
        return 'bash ./kge/kge.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {} -de -dr -r {}'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk, kge_reg)

def cmd_mln(main_path, workspace_path=None, preprocessing=False):
    if preprocessing == True:
        return './mln/mln -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(main_path, main_path, main_path, mln_threshold_of_rule, mln_threads)
    else:
        return './mln/mln -load {}/mln_saved.txt -probability {}/annotation.txt -out-prediction {}/pred_mln.txt -out-rule {}/rule.txt -thresh-triplet 1 -iterations {} -lr {} -threads {}'.format(main_path, workspace_path, workspace_path, workspace_path, mln_iters, mln_lr, mln_threads)

def save_cmd(save_path):
    with open(save_path, 'w') as fo:
        fo.write('dataset: {}\n'.format(dataset))
        fo.write('iterations: {}\n'.format(iterations))
        fo.write('kge_model: {}\n'.format(kge_model))
        fo.write('kge_batch: {}\n'.format(kge_batch))
        fo.write('kge_neg: {}\n'.format(kge_neg))
        fo.write('kge_dim: {}\n'.format(kge_dim))
        fo.write('kge_gamma: {}\n'.format(kge_gamma))
        fo.write('kge_alpha: {}\n'.format(kge_alpha))
        fo.write('kge_lr: {}\n'.format(kge_lr))
        fo.write('kge_iters: {}\n'.format(kge_iters))
        fo.write('kge_tbatch: {}\n'.format(kge_tbatch))
        fo.write('kge_reg: {}\n'.format(kge_reg))
        fo.write('mln_threshold_of_rule: {}\n'.format(mln_threshold_of_rule))
        fo.write('mln_threshold_of_triplet: {}\n'.format(mln_threshold_of_triplet))
        fo.write('mln_iters: {}\n'.format(mln_iters))
        fo.write('mln_lr: {}\n'.format(mln_lr))
        fo.write('mln_threads: {}\n'.format(mln_threads))
        fo.write('weight: {}\n'.format(weight))

time = str(datetime.datetime.now()).replace(' ', '_')
path = path + '/' + time
ensure_dir(path)
save_cmd('{}/cmd.txt'.format(path))

# ------------------------------------------

os.system('cp {}/train.txt {}/train.txt'.format(dataset, path))
os.system('cp {}/train.txt {}/train_augmented.txt'.format(dataset, path))
os.system(cmd_mln(path, preprocessing=True))

for k in range(iterations):

    workspace_path = path + '/' + str(k)
    ensure_dir(workspace_path)

    os.system('cp {}/train_augmented.txt {}/train_kge.txt'.format(path, workspace_path))
    os.system('cp {}/hidden.txt {}/hidden.txt'.format(path, workspace_path))
    os.system(cmd_kge(workspace_path, kge_model))

    os.system(cmd_mln(path, workspace_path, preprocessing=False))
    augment_triplet('{}/pred_mln.txt'.format(workspace_path), '{}/train.txt'.format(path), '{}/train_augmented.txt'.format(workspace_path), mln_threshold_of_triplet)
    os.system('cp {}/train_augmented.txt {}/train_augmented.txt'.format(workspace_path, path))

    evaluate('{}/pred_mln.txt'.format(workspace_path), '{}/pred_kge.txt'.format(workspace_path), '{}/result_kge_mln.txt'.format(workspace_path), weight)
