import numpy as np
import time, math
import tensorflow as tf


def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=1, pixel_level=False):
  """ It returns eraser function to apply cutout on input image
    Parameters:
    p   : probability of cutout application
    s_l : min cutout area proportion
    s_h : max cutout area proportion
    r_1 : min aspect ratio
    r_2 : max aspect ratio
    v_l : min value for erased area
    v_h : max value for erased area
    pixel_level(bool) : whether to apply random pixel values in cutout area
    Returns:
    eraser function which applies cutout to the input image.
  """

  def eraser(input_img: tf.Tensor) -> tf.Tensor:
      dtype = input_img.dtype
      inp_shape = tf.shape(input_img)
      img_h = inp_shape[0]
      img_w = inp_shape[1]
      img_c = inp_shape[2]
      
      p_1 = tf.random.uniform(shape = [],dtype=tf.float32)

      if p_1 > p:
          return input_img

      
      s = tf.random.uniform(shape=[],minval=s_l,maxval=s_h,dtype=tf.float32) * tf.cast(img_h,tf.float32) * tf.cast(img_w,tf.float32)
      r = tf.random.uniform(shape=[],minval=r_1,maxval=r_2,dtype=tf.float32)
      w = tf.cast(tf.math.sqrt(s / r),tf.int32)
      h = tf.cast(tf.math.sqrt(s * r),tf.int32)
      left = tf.random.uniform(shape=[],minval=0,maxval=img_w,dtype=tf.int32)
      top = tf.random.uniform(shape=[],minval=0,maxval=img_h,dtype=tf.int32) 

      if left + w > img_w:
        w = img_w - left - 1
      if top + h > img_h:
        h = img_h - top -  1

      replaceBlock = tf.random.uniform([h, w, img_c], minval=v_l, maxval=v_h, dtype=dtype)           
      
      input_img = replace_slice(input_img, replaceBlock, [top, left, 0])
      return input_img

  return eraser
