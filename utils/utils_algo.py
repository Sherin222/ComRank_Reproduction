import math
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import numpy as np
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rank_exp_consis(outputs, com_labels):#consis exp version
    # Apply sigmoid activation to convert logits to probabilities
    sig_outputs = torch.sigmoid(outputs)

    # Get scores for the complementary labels
    comp_scores = torch.sum(sig_outputs * com_labels, dim=1, keepdim=True)

    # Compute the differences exp(f_u(x_i) - f_y_bar(x_i)) for all u != y_bar
    differences = sig_outputs - comp_scores.expand_as(sig_outputs)

    # Compute the exponential terms
    exp_terms = torch.exp(-differences)

    # Mask out the complementary labels to exclude them from the sum
    non_com_labels = 1 - com_labels
    masked_exp_terms = exp_terms * non_com_labels

    # Compute the log-sum-exp over the non-complementary labels
    sample_loss =torch.sum(masked_exp_terms, dim=1)

    # Calculate the mean of the loss over the batch
    loss = torch.mean(sample_loss)

    return loss
def rank_exp_consis_high(outputs, com_labels):
    """
    Custom ranking loss implementation for multiple complementary labels for batch processing.

    Args:
    sig_outputs: A tensor of shape (batch_size, num_classes) representing the raw output logits from the model.
    com_labels: A binary tensor of shape (batch_size, num_classes) where 1s indicate complementary labels.

    Returns:
    loss: Scalar tensor representing the average loss over the batch.
    """
    sig_outputs = torch.sigmoid(outputs)
    # Create masks for complementary and non-complementary labels
    comp_mask = com_labels == 1
    non_comp_mask = com_labels == 0

    # Calculate the tensor of differences
    diffs = sig_outputs.unsqueeze(2) - sig_outputs.unsqueeze(1)
    exp_diffs = torch.exp(-diffs)

    # Apply complementary and non-complementary masks
    # This zeros out differences that do not involve a non-comp and comp pair
    relevant_diffs = exp_diffs * non_comp_mask.unsqueeze(2) * comp_mask.unsqueeze(1)
    # Compute log(1+x) for summed differences

    log_terms = torch.log1p(relevant_diffs)

    # Sum up all the relevant differences across complementary labels for each non-complementary label
    summed_log_terms = torch.sum(log_terms, dim=2)

    # Sum over all non-complementary labels for each sample and then take the mean across the batch
    sample_losses = torch.sum(summed_log_terms * non_comp_mask, dim=1)
    mean_loss = torch.mean(sample_losses)

    return mean_loss
#################################################################
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def predict(Outputs, threshold):
    sig = nn.Sigmoid()
    # pre = sig(Outputs)
    pre_label = sig(Outputs)
    min_pre_label, _ = torch.min(pre_label.data, 1)
    max_pre_label, _ = torch.max(pre_label, 1)
    min_pre_label = min_pre_label.view(-1, 1)
    max_pre_label = max_pre_label.view(-1, 1)
    for i in range(len(pre_label)):
        if max_pre_label[i] - min_pre_label[i] != 0:
            pre_label[i, :] = (pre_label[i, :] - min_pre_label[i]) / (max_pre_label[i] - min_pre_label[i])
    pre = pre_label
    # pre[pre > 0.45] = 1
    # pre[pre <= 0.45] = 0
    pre[pre > threshold] = 1
    pre[pre <= threshold] = 0
    return pre

#####################################################################################################
def gdf(outputs, com_labels):
    '''
    for a drive loss based the bce loss, where l means -log or others loss function.
    The loss is an upper-bound of loss_unbiase_1.
    :param outputs: the presicted value with n*K size
    :param com_labels: the complementary label matrix, which is a one-hot vector for per instance
    :return: the loss value
    '''
    n, K = com_labels.size()[0], com_labels.size()[1]
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    # part_1 = -torch.log(torch.sum(pos_outputs*sig_outputs, dim=1) + 1e-12).mean()
    # part_3 = -torch.log(torch.sum(neg_outputs * (1.0 - sig_outputs), dim=1) + 1e-12).mean()
    part_1 = -torch.sum(torch.log(sig_outputs + 1e-12) * pos_outputs, dim=1).mean()
    part_3 = -torch.sum(torch.log(1.0 - sig_outputs + 1e-12) * neg_outputs, dim=1).mean()
    # ave_loss = 1.0/(2**(K-1) - 1.0) * part_1 + part_3
    ave_loss = part_1 + part_3
    return ave_loss

def mae(outputs, com_labels):

    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sig_outputs, pos_outputs.float())
    sample_loss = loss_matrix.sum(dim=-1).mean()
    return sample_loss
def unbiased_bce(outputs, com_labels):
    '''
    for a drive loss based the bce loss, where l means -log or others loss function
    :param outputs: the presicted value with n*K size
    :param com_labels: the complementary label matrix, which is a one-hot vector for per instance
    :return: the loss value
    '''
    n, K = com_labels.size()[0], com_labels.size()[1]
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    part_1 = -torch.sum(pos_outputs * torch.log(sig_outputs + 1e-12), dim=1).mean()#取-号是应为二分类损失函数有-号
    part_2 = -torch.sum(pos_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1).mean()
    part_3 = -torch.sum(neg_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1).mean()
    ave_loss = 2**(K-2)/(2**(K-1)-1) * part_1 + (2**(K-2) - 1)/(2**(K-1)-1) * part_2 + part_3
    return ave_loss
