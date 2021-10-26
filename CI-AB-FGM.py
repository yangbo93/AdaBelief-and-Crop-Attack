"""Implementation of sample attack on Inception_v3"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image
#from scipy.misc import imread, imresize, imsave
#from scipy.misc import imresize
import imageio
from pylab import *

import time
import tensorflow as tf
import scipy.stats as st
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

#delete all of flags before running the main command     
#del_all_flags(tf.flags.FLAGS)
        


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', './models/inception_v3.ckpt', 'Path to checkpoint for inception network.')

# =============================================================================
# tf.flags.DEFINE_string(
#     'checkpoint_path', './models/inception_v4.ckpt', 'Path to checkpoint for inception network.')
# 
# tf.flags.DEFINE_string(
#     'checkpoint_path', './models/resnet_v2_101.ckpt', 'Path to checkpoint for inception network.')
# 
# tf.flags.DEFINE_string(
#     'checkpoint_path', './models/inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')
# =============================================================================

tf.flags.DEFINE_string(
   'input_dir', './dataset/images', 'Input directory with images.')

tf.flags.DEFINE_string(
   'output_dir', './output_test', 'Output directory with images.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_float(
    'prob', 1.0, 'probability of using diverse inputs.')

# if momentum = 1, this attack becomes M-DI-2-FGSM
tf.flags.DEFINE_float(
    'momentum', 1, 'Momentum.')

tf.flags.DEFINE_string(
    'GPU_ID', '0,1', 'which GPU to use.')

se = int(time.time())
print (se)
np.random.seed(se)
tf.set_random_seed(se)

print("print all settings\n")
print(FLAGS.master)
print(FLAGS.__dict__)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_ID


def load_images(input_dir, output_dir, batch_shape):
  """Read png images from input directory in batches.
  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png'))[:1000]:
    temp_name = str.split(filepath, '/')
    output_name = output_dir + '/'+ temp_name[-1]
    # check if the file exist
    if os.path.isfile(output_name) == False:
#      with tf.gfile.Open(filepath) as f:
      with tf.io.gfile.GFile(filepath, "rb") as f:
        image = imageio.imread(f, pilmode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
      images[idx, :, :, :] = image * 2.0 - 1.0
      filenames.append(os.path.basename(filepath))
      idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.
  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
     # imageio.imsave(f, (images[i, :, :, :] + 1.0) * 0.5 * 255, format='png')
      imageio.imsave(f, Image.fromarray(uint8((images[i, :, :, :] + 1.0) * 0.5 * 255)), format='png')



beta1 = 0.9
beta2 = 0.999
num_iter1 = FLAGS.num_iter
weight=0
t = np.arange(1,num_iter1+0.1,1)
y1 = np.sqrt(1 - beta2**t) / (1 - beta1**t)

for x1 in y1:
    weight+=x1
      

def graph(x, y, i, x_max, x_min, grad, grad2):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  num_iter = FLAGS.num_iter
  batch_size = FLAGS.batch_size
  alpha = eps / num_iter
  alpha_norm2 = eps * np.sqrt(299 * 299 * 3) / num_iter
  momentum = FLAGS.momentum

  delta = 1e-14
  num_classes = 1001
  x_nes = x #MI
  #x_nes = x + momentum * alpha * grad  #NI
#====================================v3=========================================
  
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, end_points = inception_v3.inception_v3(
        Crop(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)    

  pred = tf.argmax(end_points['Predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits)
  auxlogits = end_points['AuxLogits']   
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)
  # compute the gradient info 
  noise = tf.gradients(cross_entropy, x_nes)[0]
  
  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits2, end_points2 = inception_v3.inception_v3(
      Crop(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
  
  pred = tf.argmax(end_points2['Predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits2)
  auxlogits = end_points2['AuxLogits']   
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)
  # compute the gradient info 
  noise2 = tf.gradients(cross_entropy, x_nes)[0]


  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits3, end_points3 = inception_v3.inception_v3(
      Crop(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
  
  pred = tf.argmax(end_points3['Predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits3)
  auxlogits = end_points3['AuxLogits']   
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)
  # compute the gradient info 
  noise3 = tf.gradients(cross_entropy, x_nes)[0]  
  

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits4, end_points4 = inception_v3.inception_v3(
      Crop(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
  
  pred = tf.argmax(end_points4['Predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits4)
  auxlogits = end_points4['AuxLogits']   
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)
  # compute the gradient info 
  noise4 = tf.gradients(cross_entropy, x_nes)[0]
  

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits5, end_points5 = inception_v3.inception_v3(
      Crop(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
  
  pred = tf.argmax(end_points5['Predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits5)
  auxlogits = end_points5['AuxLogits']   
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)
  # compute the gradient info 
  noise5 = tf.gradients(cross_entropy, x_nes)[0]
   

  noise = (noise+noise2+noise3+noise4+noise5) / 5
  
  
#=============================== AdaBelief==================================  
  noise2 = grad2
  noise = noise / tf.reduce_sum(tf.abs(noise), [1,2,3], keep_dims=True)
  noise_gt = noise
  
  noise = beta1 * grad + (1-beta1) * noise
  noise2 = beta2 * grad2 + (1-beta2) * tf.square(noise_gt-noise)
  
  noise_t = tf.divide(noise,1-beta1**(i+1))
  noise2_t = tf.divide(noise2+delta,1-beta2**(i+1))
  
  normalized_grad = noise_t/(tf.sqrt(noise2_t)+delta)
  square = tf.reduce_sum(tf.square(normalized_grad),reduction_indices=[1,2,3],keep_dims=True)
  normalized_grad = normalized_grad / tf.sqrt(square)
  x = x + alpha_norm2 * normalized_grad


#===========================MI==================================================  
# =============================================================================
#   noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
#   noise = momentum * grad + noise
#   x = x + alpha * tf.sign(noise)
# =============================================================================
    
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise, noise2


def stop(x, y, i, x_max, x_min, grad, grad2):
  #num_iter = int(min(FLAGS.max_epsilon+4, 1.25*FLAGS.max_epsilon))
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)


def Crop(input_tensor):
    rnd = tf.random_uniform((), 279, 299, dtype=tf.int32)
    rescaled = tf.image.resize_image_with_crop_or_pad(input_tensor, rnd, rnd)
    h_rem = 299 - rnd
    w_rem = 299 - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], 299, 299, 3))
    return tf.cond(tf.random.uniform([], 0, 1) < 1.0, lambda: padded, lambda: input_tensor)

def main(_):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    i = tf.constant(0,float)
    grad = tf.zeros(shape=batch_shape)
    grad2 = tf.zeros(shape=batch_shape)
    x_adv, pre, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad, grad2])
    # Run computation
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, FLAGS.checkpoint_path)
      for filenames, images in load_images(FLAGS.input_dir, FLAGS.output_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)

if __name__ == '__main__':
  tf.app.run()




