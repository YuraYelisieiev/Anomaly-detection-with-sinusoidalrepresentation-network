import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        distance_positive = 1 - cos(anchor, positive)
        distance_negative = 1 - cos(anchor, negative)
        # distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        # distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def gram_matrix(input_features):
    a, b, c, d = input_features.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input_features.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
