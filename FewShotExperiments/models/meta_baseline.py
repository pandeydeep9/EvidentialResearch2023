import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):
    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        # print("6. X_shot: ", x_shot.shape) #6. X_shot:  torch.Size([4, 5, 1, 3, 80, 80])
        # print("7. X_query: ", x_query.shape)#6. X_shot:  torch.Size([75, 1, 3, 80, 80])
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1) #Embedding of support set 4*5*Em
        x_query = x_query.view(*query_shape, -1) #Embedding of query set 75*Em

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2) #Across class average prototype
            x_shot = F.normalize(x_shot, dim=-1) #Normalize across prototypes within a batch
            x_query = F.normalize(x_query, dim=-1) #Normalize across the query set instances
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2) #Across class average prototype
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits

