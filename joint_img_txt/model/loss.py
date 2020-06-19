'''
Authors: Geeticka Chauhan, Ruizhi Liao

Custom loss functions for the joint model
'''
import random

import torch
from torch.nn import CosineSimilarity, MarginRankingLoss


def ranking_loss(z_image, z_text, y, report_id, 
                 similarity_function='dot'):
    """
    A custom ranking-based loss function
    Args:
        z_image: a mini-batch of image embedding features
        z_text: a mini-batch of text embedding features
        y: a 1D mini-batch of image-text labels 
    """
    return imposter_img_loss(z_image, z_text, y, report_id, similarity_function) + \
           imposter_txt_loss(z_image, z_text, y, report_id, similarity_function)

def imposter_img_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the image is an imposter image chosen from the batch 
    """
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)

    for i in range(batch_size):
        if similarity_function == 'dot':
            paired_similarity = torch.dot(z_image[i], z_text[i])
        if similarity_function == 'cosine':
            paired_similarity = \
                torch.dot(z_image[i], z_text[i])/(torch.norm(z_image[i])*torch.norm(z_text[i]))
        if similarity_function == 'l2':
            paired_similarity = -1*torch.norm(z_image[i]-z_text[i])

        # Select an imposter image index and 
        # compute the maximum margin based on the image label difference
        j = i+1 if i < batch_size - 1 else 0
        if report_id[i] == report_id[j]: 
        # This means the imposter image comes from the same acquisition 
            margin = 0
        elif y[i].item() == -1 or y[j].item() == -1: # '-1' means unlabeled 
            margin = 0.5
        else:
            margin = max(0.5, (y[i] - y[j]).abs().item())

        if similarity_function == 'dot':
            imposter_similarity = torch.dot(z_image[j], z_text[i])
        if similarity_function == 'cosine':
            imposter_similarity = \
                torch.dot(z_image[j], z_text[i])/(torch.norm(z_image[j])*torch.norm(z_text[i]))
        if similarity_function == 'l2':
            imposter_similarity = -1*torch.norm(z_image[j]-z_text[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        if diff_similarity > 0:
            loss = loss + diff_similarity

    return loss / batch_size # 'mean' reduction

def imposter_txt_loss(z_image, z_text, y, report_id, similarity_function):
    """
    A custom loss function for computing the hinge difference 
    between the similarity of an image-text pair and 
    the similarity of an imposter image-text pair
    where the text is an imposter text chosen from the batch 
    """
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    batch_size = z_image.size(0)

    for i in range(batch_size):
        if similarity_function == 'dot':
            paired_similarity = torch.dot(z_image[i], z_text[i])
        if similarity_function == 'cosine':
            paired_similarity = \
                torch.dot(z_image[i], z_text[i])/(torch.norm(z_image[i])*torch.norm(z_text[i]))
        if similarity_function == 'l2':
            paired_similarity = -1*torch.norm(z_image[i]-z_text[i])

        # Select an imposter text index and 
        # compute the maximum margin based on the image label difference
        j = i+1 if i < batch_size - 1 else 0
        if report_id[i] == report_id[j]: 
            # This means the imposter report comes from the same acquisition 
            margin = 0
        elif y[i].item() == -1 or y[j].item() == -1: # '-1' means unlabeled
            margin = 0.5
        else:
            margin = max(0.5, (y[i] - y[j]).abs().item())

        if similarity_function == 'dot':
            imposter_similarity = torch.dot(z_text[j], z_image[i])
        if similarity_function == 'cosine':
            imposter_similarity = \
                torch.dot(z_text[j], z_image[i])/(torch.norm(z_text[j])*torch.norm(z_image[i]))
        if similarity_function == 'l2':
            imposter_similarity = -1*torch.norm(z_text[j]-z_image[i])

        diff_similarity = imposter_similarity - paired_similarity + margin
        if diff_similarity > 0:
            loss = loss + diff_similarity

    return loss / batch_size # 'mean' reduction

def dot_product_loss(z_image, z_text):
    batch_size = z_image.size(0)
    loss = torch.zeros(1, device=z_image.device, requires_grad=True)
    for i in range(batch_size):
        loss = loss - torch.dot(z_image[i], z_text[i])
    return loss / batch_size

