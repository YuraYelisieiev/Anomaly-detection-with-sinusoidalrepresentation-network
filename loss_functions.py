import torch
import torch.nn.functional as F
import diff_operators
import torch.nn as nn
import dataio


def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}


def image_mae(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}


def gradients_mse(model_output, gt):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])

    gradients_loss = torch.mean((gradients[:, :, 0:2] - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def laplace_mse(model_output, gt):
    laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    laplace_loss = torch.mean((laplace[:, :, 0:2] - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}


def second_order_loss(mask, model_output, gt):
    grad = gradients_mse(model_output, gt)['gradients_loss']
    laplace = laplace_mse(model_output, gt)['laplace_loss']

    image_loss = ((model_output['model_out'] - gt['img']) ** 2).mean()

    final_loss = (grad + laplace + image_loss) / 3

    return {"cmb_loss": final_loss}
