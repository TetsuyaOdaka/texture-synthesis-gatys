'''
    Texture Synthesis Using Convolutional Neural Networks
	Gatys et al. (2015) aiXiv
    https://arxiv.org/abs/1505.07376

    python generate_texture.py -w 600 --iter 3000 -s images/pebble.png -o /home/odaka/out-texture -g 0

    -w : width of image : squared images are assumed for both content and style images

    -s ： path it style image
    -o ： base directory

    -g ： GPU number -1: not use (attention: not tested)
    -v ： 1: create images in the middle of process (by 100 iteration)

    --iter : iteration
    
'''
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import os
import sys
import math
import numpy as np
from numpy import random
from PIL import Image

import argparse
import logging
from datetime import datetime
import time

SCRIPT_NAME = os.path.basename(__file__)
# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"

#logging.basicConfig(level=logging.INFO, filename=FPATH, format=LOG_FMT)
#STDATA = datetime.now()
#FNAME = 'deep-art-gatys-texture-' + STDATA.strftime('%Y%m%d') + '.log'
#FPATH = 'logs/{0}'.format(FNAME)

logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

# Set of Layers : Gatys et al.(2015)
#STYLE_LAYERS = ["conv1_1"]
#STYLE_LAYERS = ["conv1_1", "pool1"]
#STYLE_LAYERS = ["conv1_1", "pool1", "pool2"]
#STYLE_LAYERS = ["conv1_1", "pool1", "pool2", "pool3"]
STYLE_LAYERS = ["conv1_1", "pool1", "pool2", "pool3", "pool4"]	
#STYLE_LAYERS = ["conv1_1", "pool1", "pool2", "pool3", "pool4", "pool5"]	

# Gatys Net
class GatysNet(L.VGG16Layers):
	def __init__(self):
		super(GatysNet, self).__init__()
		with self.init_scope():
			self.model = L.VGG16Layers()

	def __call__(self, x):
		#conv1
		y1 = self.model.conv1_2(F.relu(self.model.conv1_1(x)))
		x0 = self.model.conv1_1(x)
		x1 = F.average_pooling_2d(F.relu(y1), 2, stride=2)
		#conv2
		y2 = self.model.conv2_2(F.relu(self.model.conv2_1(x1)))
		x2 = F.average_pooling_2d(F.relu(y2), 2, stride=2)
		#conv3
		y3 = self.model.conv3_3(F.relu(self.model.conv3_2(F.relu(self.model.conv3_1(x2)))))
		x3 = F.average_pooling_2d(F.relu(y3), 2, stride=2)
		#conv4
		y4 = self.model.conv4_3(F.relu(self.model.conv4_2(F.relu(self.model.conv4_1(x3)))))
		x4 = F.average_pooling_2d(F.relu(y4), 2, stride=2)
		#conv5
		y5 = self.model.conv5_3(F.relu(self.model.conv5_2(F.relu(self.model.conv5_1(x4)))))
		x5 = F.average_pooling_2d(F.relu(y5), 2, stride=2)

		return {'conv1_1': x0, 'pool1': x1, 'pool2': x2, 'pool3': x3, 'pool4': x4, 'pool5' : x5}

# Generate Texture
class GenTexture(chainer.Link):

	def __init__(self, img_style):
		super(GenTexture, self).__init__()

		# feature maps of style image created by Gatys Net
		sw = img_style.shape[0]
		sh = img_style.shape[1]
		img_style = chainer.Variable(img_style.transpose(2,0,1).reshape(1, 3, sw, sh))
		vgg = ga_model(img_style)
		# Gram Matrix of style image
		self.style_matrix = self.get_matrix(vgg)
		
		# initialize output image by uniform distributed random values
		if args.gpu >= 0:
			w = xp.random.uniform(-20, 20, (1, 3, sw, sh), dtype=np.float32)
		else:
			w = np.random.uniform(-20, 20, (1, 3, sw, sh)).astype(np.float32)
		
		with self.init_scope():
			# this model holds only chainer.Parameter
			self.W = chainer.Parameter(w)

	# calicurate gram matrix
	def get_matrix(self, vgg):
		result = []
		for i in STYLE_LAYERS:
			# channel
			ch = vgg[i].data.shape[1]
			# width (= height)
			wd = vgg[i].data.shape[2]
			# feature matrix : F in paper.
			fm = F.reshape(vgg[i], (ch,wd**2))
			# G = F * t(F)
			# G = G / Ml**2
			result.append(F.matmul(fm, fm, transb=True)/wd**2)
		return result

	def __call__(self):
		# get Gatys Net output of content and style matrix
		vgg = ga_model(self.W)
		gen_matrix = self.get_matrix(vgg)
		# calicurate loss
		loss = 0
		# style loss
		for i in range(len(STYLE_LAYERS)):
			loss += F.mean_squared_error(gen_matrix[i], self.style_matrix[i]) / xp.float32(4.)

		return loss


# resize image
def image_resize(img_file, width):
	# PIL image
	pil_im = Image.open(img_file).convert('RGB')
	orig_w, orig_h = pil_im.size[0], pil_im.size[1]
	if orig_w > orig_h:
		new_w = width
		new_h = width * orig_h // orig_w
	else:
		new_w = width * orig_w//orig_h
		new_h = width
	
	pil_im = pil_im.resize((new_w, new_h))
	pil_im = xp.asarray(pil_im, dtype=np.float32).reshape(new_w, new_h, 3)

	# https://github.com/cupy/cupy/issues/589
	data = np.zeros((new_w, new_h, 3)).astype(np.float32)
	im = pil_im
	if args.gpu >= 0:
		im = chainer.cuda.to_cpu(im)

	# subtribe mean values of VGG model
	# Don't Clip Here!!
	data[:,:,0] = im[:,:,2] - 123.68 # B
	data[:,:,1] = im[:,:,1] - 116.779 # G
	data[:,:,2] = im[:,:,0] - 103.939 # R
	if args.gpu >= 0:
		data = chainer.cuda.to_gpu(data)
	
	LOGGER.info("image resized to: {}".format(str(pil_im.shape[0])))
    
	return data, new_w, new_h


# save image
def save_image(width, height, idx, dir_name):
	data = np.zeros((width, height, 3), dtype=np.uint8)
	tmp_im = model.W.data[0]  # BGR
	if args.gpu >= 0:
		tmp_im = chainer.cuda.to_cpu(tmp_im)

	# add mean value and transpose BGR to RGB
	data[:,:,0] = (tmp_im[2] + 103.939).clip(0,255) # R
	data[:,:,1] = (tmp_im[1] + 116.779).clip(0,255) # G
	data[:,:,2] = (tmp_im[0] + 123.68).clip(0,255) # B
	himg = Image.fromarray(data, 'RGB')
	himg.save(args.out_dir + "/" + dir_name + "/" + dir_name + "_im_{}.jpg".format(str(idx)))
	return

LOGGER.info('script start')
start_time = time.time()

parser = argparse.ArgumentParser(
    description='Texture Synthesis Using Convolutional Neural Networks')
parser.add_argument('--style_img', '-s', default='style.png',
                    help='Style image')
parser.add_argument('--out_dir', '-o', default='output',
                    help='Output directory')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--iter', default=3000, type=int,
                    help='number of iteration')
parser.add_argument('--lr', default=4.0, type=float,
                    help='learning rate')
parser.add_argument('--width', '-w', default=435, type=int,
                    help='image width, height')
parser.add_argument('--verbose', '-v', default=1, type=int,
                    help='verbose')
args = parser.parse_args()


# name of style
sn = args.style_img.split('/')[-1].split('.')[0]
# directory name
dn = 'texture_' + sn + '_' + STYLE_LAYERS[-1] + '_vgg_' + "{:.2f}".format(args.lr) + '_' + str(args.width)

# create directory for output
try:
    os.mkdir(args.out_dir)
except:
    pass

try:
    os.mkdir(args.out_dir + '/' + dn)
except:
    pass


# GPU/CPU
if args.gpu >= 0:
	import cupy as xp
	import chainer.cuda as cuda
	cuda.check_cuda_available()
	F.type_check_enable = False
	cuda.get_device(args.gpu).use()
else:
    xp = np

# initialize Gatys Net
ga_model = GatysNet()
if args.gpu >= 0:
	ga_model.to_gpu()

# resize content and style images
W = args.width
img_style, nw, nh = image_resize(args.style_img, W)


# initialize model
model = GenTexture(img_style)
if args.gpu >= 0:
	model.to_gpu()

# create chainer.Optimizer
optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)

for i in range(0, args.iter):
	model.zerograds()

	# forward
	Loss = model()

	# BP
	Loss.backward()

	# update
	optimizer.update()

	if i%100==0:
		LOGGER.info('{} {}'.format(str(i), str(Loss.data)))
		if args.verbose==1:
			# save image
			save_image(img_style.shape[0], img_style.shape[1], i, dn)

# save image
save_image(img_style.shape[0], img_style.shape[1], i, dn)

elapsed_time = time.time() - start_time
LOGGER.info('script end')
LOGGER.info('elapsed time : {} [sec]'.format(str(elapsed_time)))
