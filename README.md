# Neural Texture Synthesis by Gatys et al. Chainer 4.0 
[Chainer v4.0](https://github.com/chainer/chainer) implementation of "Texture Synthesis Using Convolutional Neural Networks(2015)" by Gatys et al.

[Texture Synthesis Using Convolutional Neural Networks. Gatys, L.A. et al(2015)](https://arxiv.org/pdf/1505.07376.pdf) 


 The purpose is further research of amazing Gatsy's article. 
 So source codes are simple for modification and for that reason, sophisticated functions of chainer(i.e. updater, trainer) are not used.


 ## Results
 ### Original Image
 pebbles. http://www.cns.nyu.edu/~lcv/texture/color/
 
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-gatys/blob/master/images/pebble.jpg" width="256" alt="pebbles"> 

 ### Synthesized Image
 
<img src="https://github.com/TetsuyaOdaka/texture-synthesis-gatys/blob/master/samples/texture_pebble_pool4_vgg_4.00_256_im_2999.jpg" width="256" alt="synthesizeed pebbles"> 



 ## 
 GPU: GeForce GTX1080i 
 
 Elapsed Time: about 100sec for generating 256px squared image. 
 
 
 
# Usage 
## Environment
- python3.5 (3.0+)
- chainer4.0
- cupy4.0.0
- cuda8.0+
- cuDNN6.0+

## Generate Transferred Image 
`python generate_texture.py -w 256 --iter 3000 -s [dir_path]/pebble.jpg -o [dir_path for output] -g 0` 
 
 **Shape of images must be square.**

## Parameters
See [source](https://github.com/TetsuyaOdaka/texture-synthesis-gatys/blob/master/generate_texture.py)


# Acknowlegement
thanks to the authors to which I referred.
- chainer-gogh(https://github.com/pfnet-research/chainer-gogh) 
