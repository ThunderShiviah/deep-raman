import skimage
from skimage.metrics import structural_similarity as ssim
import numpy as np
import tensorflow as tf

def _ssim_loss(y_true, y_pred):
  return tf.reduce_mean(-1*tf.image.ssim(y_true, y_pred, max_val=1))

def psnr_loss(y_true, y_pred):
  return tf.reduce_mean(-1.0*tf.image.psnr(y_true, y_pred, 1))
