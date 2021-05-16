import torch
from torch import nn
import pytorch_lightning as pl
import dataio
from torchvision.models import vgg16
from models import SingleBVPNet, Encoder, Discriminator, TripletNet, LossNetwork
from losses import TripletLoss
from functools import partial
import utils
import loss_functions
from copy import deepcopy

class Reconstruction(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        #         If you want to manually write optimization process uncomment
        #         self.automatic_optimization = False
        self.generator = Generator(config)
        vgg_model = vgg16(pretrained=True)
        self.p_loss_net = LossNetwork(vgg_model)
        self.p_loss_net.eval()

        self.config = config
        self.image_resolution = config['data']['after_crop_size'] // 2
        self.steps_till_summary = config['model']['steps_till_summary']
        center = self.image_resolution
        hs = self.image_resolution // 2

        self.crop_from = center - hs
        self.crop_to = center + hs

        self.g_loss_w = config['model']['generator']['loss_w']
        self.mse_w = config['model']['generator']['mse_weight']
        self.perc_w = config['model']['generator']['perc_weight']
        self.gram_w = config['model']['generator']['gram_weight']
        self.d_loss_w = config['model']['discriminator']['loss_w']

        self.g_loss = partial(loss_functions.image_mse, None)
        self.summary_fn = partial(utils.write_image_summary, self.image_resolution)
        self.mse_criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        g_loss, mse_loss, perceptual, gram_loss = self.shared_step(batch, 'train_')
        self.log('train/GenLoss', g_loss)
        self.log('train/MseLoss', mse_loss)
        self.log('train/perceptual', perceptual)
        self.log('train/GramLoss', gram_loss)
        return g_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        g_loss, mse_loss, perceptual, gram_loss = self.shared_step(batch, 'val_')
        self.log('val/GenLoss', g_loss)
        self.log('val/MseLoss', mse_loss)
        self.log('val/perceptual', perceptual)
        self.log('val/GramLoss', gram_loss)
        return g_loss

    def shared_step(self, batch, prefix):
        model_input, gt = batch

        generator_out = self.generator(model_input)

        gt_img, pred_img, _, _ = self.prepare_images(gt, model_input, generator_out)

        gt_full_emb = self.p_loss_net(gt_img.repeat(1, 3, 1, 1))
        pred_full_emb = self.p_loss_net(pred_img.repeat(1, 3, 1, 1))

        perceptual = self.perceptual_loss(gt_full_emb, pred_full_emb)

        img_mse = self.generator_loss(generator_out, gt)

        gram_loss = self.style_loss(gt_full_emb, pred_full_emb)

        g_loss = self.mse_w * img_mse + self.perc_w * perceptual + self.gram_w * gram_loss

        if (not self.global_step % self.steps_till_summary) or (
                prefix == "val_" and self.global_step % self.steps_till_summary < 5):
            self.summary_fn(self.generator,
                            model_input,
                            gt,
                            generator_out,
                            self.logger.experiment,
                            self.global_step,
                            prefix=prefix)

        return g_loss, img_mse, perceptual, gram_loss

    def generator_loss(self, generator_out, gt):
        return self.g_loss(generator_out, gt)['img_loss']

    def configure_optimizers(self):
        optim_g = torch.optim.Adam(lr=self.config['model']['generator']['learning_rate'],
                                   params=self.generator.parameters())
        return optim_g

    def perceptual_loss(self, features, targets):
        content_loss = 0
        for f, t, in zip(features, targets):
            content_loss += self.mse_criterion(f, t)
        return content_loss

    def gram(self, x):
        b, c, h, w = x.size()
        g = torch.bmm(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2))
        return g.div(h * w)

    def style_loss(self, features, targets):
        gram_loss = 0
        for f, t in zip(features, targets):
            gram_loss += self.mse_criterion(self.gram(f), self.gram(t))
        return gram_loss

    def prepare_images(self, gt, model_input, generator_out):
        gt_img = dataio.lin2img(gt['img'], self.image_resolution)
        pred_img = dataio.lin2img(generator_out['model_out'], self.image_resolution)
        cropped_img = model_input['border']

        gt_big_img = deepcopy(cropped_img)
        pred_big_img = deepcopy(cropped_img)

        gt_big_img[:, :, self.crop_from: self.crop_to, self.crop_from: self.crop_to] = gt_img

        pred_big_img[:, :, self.crop_from: self.crop_to, self.crop_from: self.crop_to] = torch.clamp(pred_img, 0, 255)

        return gt_big_img, pred_big_img, gt_img, pred_img


class RGAN(Reconstruction):
    def __init__(self, config):
        super().__init__(config)
        embedding_net = Discriminator()
        self.discriminator = TripletNet(embedding_net)
        self.discriminator_loss = TripletLoss(margin=0.1)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        g_loss, d_loss, mse_loss, perceptual, gram_loss = self.shared_step(batch, 'train_')
        loss_combined = self.g_loss_w * g_loss + self.d_loss_w * d_loss
        self.log('train/GenLoss', g_loss)
        self.log('train/DisLoss', d_loss)
        self.log('train/MseLoss', mse_loss)
        self.log('train/TotalLoss', loss_combined)
        self.log('train/perceptual', perceptual)
        self.log('train/GramLoss', gram_loss)
        return g_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        g_loss, d_loss, mse_loss, perceptual, gram_loss = self.shared_step(batch, 'val_')
        loss_combined = self.g_loss_w * g_loss + self.d_loss_w * d_loss
        self.log('val/GenLoss', g_loss)
        self.log('val/DisLoss', d_loss)
        self.log('val/MseLoss', mse_loss)
        self.log('val/TotalLoss', loss_combined)
        self.log('val/perceptual', perceptual)
        self.log('val/GramLoss', gram_loss)
        return g_loss

    def shared_step(self, batch, prefix):
        model_input, gt = batch

        generator_out = self.generator(model_input)

        gt_img, pred_img, _, _ = self.prepare_images(gt, model_input, generator_out)

        gt_full_emb = self.p_loss_net(gt_img.repeat(1, 3, 1, 1))
        pred_full_emb = self.p_loss_net(pred_img.repeat(1, 3, 1, 1))

        perceptual = self.perceptual_loss(gt_full_emb, pred_full_emb)

        img_mse = self.generator_loss(generator_out, gt)

        gram_loss = self.style_loss(gt_full_emb, pred_full_emb)

        g_loss = self.mse_w * img_mse + self.perc_w * perceptual + self.gram_w * gram_loss

        positive, negative, anchor = self.discriminator(model_input['rotated'], pred_img, gt_img)

        positive = torch.mean(positive, 1, True)
        negative = torch.mean(negative, 1, True)
        anchor = torch.mean(anchor, 1, True)

        d_loss = self.discriminator_loss(anchor, positive, negative)

        if (not self.global_step % self.steps_till_summary) or (
                prefix == "val_" and self.global_step % self.steps_till_summary < 5):
            self.summary_fn(self.generator,
                            model_input,
                            gt,
                            generator_out,
                            self.logger.experiment,
                            self.global_step,
                            prefix=prefix)

        return g_loss, d_loss, img_mse, perceptual, gram_loss

    def configure_optimizers(self):
        optim_g = torch.optim.Adam(lr=self.config['model']['generator']['learning_rate'],
                                   params=self.generator.parameters())
        optim_d = torch.optim.Adam(lr=self.config['model']['discriminator']['learning_rate'],
                                   params=self.discriminator.parameters())
        optimizers = [optim_d, optim_g]

        return optimizers


class Generator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.generator_config = config['model']['generator']
        self.image_resolution = config['data']['center_resolution']
        self.feature_extractor = Encoder()
        self.mlp = SingleBVPNet(act_type=self.generator_config['activation'],
                                mode=self.generator_config['mode'],
                                sidelength=self.image_resolution,
                                in_features=self.generator_config['in_features'],
                                num_hidden_layers=self.generator_config['hidden_layers'],
                                hidden_features=self.generator_config['hidden_features'])

    def forward(self, model_input):
        _, feature_number, _ = model_input['coords'].shape
        borders = self.feature_extractor(model_input['border']).unsqueeze(1)
        borders = borders.repeat(1, feature_number, 1)
        model_input['coords'] = torch.cat((model_input['coords'], borders), 2)
        generator_out = self.mlp(model_input)
        return generator_out
