import copy

import numpy as np
import torch
from PIL import Image, ImageDraw
import dataio
import os
import diff_operators
from copy import deepcopy
from torchvision.utils import make_grid
import torch.nn.functional as F
import skimage.measure
import imageio
import scipy.io.wavfile as wavfile
import cmapy


def write_image_summary(image_resolution, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_'):
    gt_img = dataio.lin2img(gt['img'], image_resolution).detach().cpu().numpy()
    pred_img = dataio.lin2img(model_output['model_out'], image_resolution).detach().cpu().numpy()
    cropped_img = model_input['border'].detach().cpu().numpy()

    center = cropped_img.shape[-1] // 2
    cv = cropped_img.shape[-1] // 4

    gt_big_img = deepcopy(cropped_img)
    pred_big_img = deepcopy(cropped_img)

    gt_big_img[:, :, center - cv: center + cv, center - cv: center + cv] = gt_img
    pred_big_img[:, :, center - cv: center + cv, center - cv: center + cv] = np.clip(pred_img, 0, 255)

    # make GIF
    if prefix == "val_":
        for idx in range(len(gt_big_img)):
            grt_img = np.rollaxis(gt_big_img[idx], 0, 3)
            new_img = np.rollaxis(pred_big_img[idx], 0, 3)
            gif_name = f"{idx}.gif"
            full_gif_path = os.path.join(writer.log_dir, gif_name)
            if not os.path.exists(full_gif_path):
                images = [np.rollaxis(gt_big_img[idx], 0, 3), np.rollaxis(pred_big_img[idx], 0, 3)]
                imageio.mimsave(full_gif_path, images, fps=1)
            else:
                old_gif_images = imageio.mimread(full_gif_path)
                old_gif_images.append(new_img)
                imageio.mimsave(full_gif_path, old_gif_images, fps=3)

    gt_img = torch.from_numpy(gt_big_img)
    pred_img = torch.from_numpy(pred_big_img)

    output_vs_gt = torch.cat((torch.from_numpy(cropped_img), gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, nrow=2, scale_each=False),
                     global_step=total_steps)

    writer.add_image(prefix + 'pred_img', make_grid(pred_img, nrow=2), global_step=total_steps)
    writer.add_image(prefix + 'gt_img', make_grid(gt_img, nrow=2), global_step=total_steps)

    write_psnr(dataio.lin2img(model_output['model_out'], image_resolution),
               dataio.lin2img(gt['img'], image_resolution), writer, total_steps, prefix + 'img_')


def write_psnr(pred_img, gt_img, writer, idx, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), idx)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), idx)


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tri_mirror(rivet, center, cv):
    left_tri_src = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(16, 16), (16, 48), (0, 32)]))
    left_tri_dst = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(16, 16), (16, 48), (32, 32)]))

    up_tri_src = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(16, 16), (32, 0), (48, 16)]))
    up_tri_dst = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(16, 16), (32, 32), (48, 16)]))

    right_tri_src = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(48, 16), (64, 32), (48, 48)]))
    right_tri_dst = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(48, 16), (32, 32), (48, 48)]))

    botom_tri_src = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(16, 48), (32, 64), (48, 48)]))
    botom_tri_dst = list(map(lambda x: (x[0] / int(16 / cv), x[1] / int(16 / cv)), [(16, 48), (32, 32), (48, 48)]))

    rivet = transformblit(left_tri_src, left_tri_dst, rivet, rivet)
    rivet = transformblit(up_tri_src, up_tri_dst, rivet, rivet)
    rivet = transformblit(right_tri_src, right_tri_dst, rivet, rivet)
    rivet = transformblit(botom_tri_src, botom_tri_dst, rivet, rivet)

    rivet = np.array(rivet)
    rivet_to_blur = rivet[center - cv: center + cv, center - cv: center + cv]
    cp_rivet = copy.deepcopy(rivet_to_blur)
    length = cv * 2
    for index in range(0, length):
        turnpixel(cp_rivet, rivet_to_blur, index, index, center)

    for index in range(0, length):
        turnpixel(cp_rivet, rivet_to_blur, index, length - index - 1, center)

    rivet[center - cv: center + cv, center - cv: center + cv] = rivet_to_blur
    rivet = Image.fromarray(rivet)
    return rivet


def transformblit(src_tri, dst_tri, src_img, dst_img):
    ((x11, x12), (x21, x22), (x31, x32)) = src_tri
    ((y11, y12), (y21, y22), (y31, y32)) = dst_tri

    M = np.array([
        [y11, y12, 1, 0, 0, 0],
        [y21, y22, 1, 0, 0, 0],
        [y31, y32, 1, 0, 0, 0],
        [0, 0, 0, y11, y12, 1],
        [0, 0, 0, y21, y22, 1],
        [0, 0, 0, y31, y32, 1]
    ])

    y = np.array([x11, x21, x31, x12, x22, x32])

    A = np.linalg.solve(M, y)

    src_copy = src_img.copy()
    srcdraw = ImageDraw.Draw(src_copy)
    srcdraw.polygon(src_tri)
    transformed = src_img.transform(dst_img.size, Image.AFFINE, A)

    mask = Image.new('1', dst_img.size)
    maskdraw = ImageDraw.Draw(mask)
    maskdraw.polygon(dst_tri, fill=255)

    dst_img.paste(transformed, mask=mask)
    return dst_img


def turnpixel(px, px2, Nix, Niy, center):
    x_start = max(0, Nix - 2)
    y_start = max(0, Niy - 2)

    x_end = Nix + 2
    y_end = Niy + 2

    if x_end >= center:
        x_end = center - 1

    if y_end >= center:
        y_end = center - 1

    for ix in range(x_start, x_end):
        for iy in range(y_start, y_end):
            def convfunc(o, v):
                return (o + int(v)) / 2

            px2[Nix, Niy] = tuple(map(convfunc, px2[Nix, Niy], px[ix, iy]))
            if sum(px2[Nix, Niy]) >= 3 * 250:
                return
