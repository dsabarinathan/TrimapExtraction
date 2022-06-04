# Trimap Extraction
This is a Keras implementation of a Light Weight CNN for Trimap Prediction. I trained Densenet based trimap prediction model. 
Natural image matting is an important problem in computer vision and graphics. It is an ill-posed problem when
only an input image is available without any external information.In practice, most existing matting methods take a
trimap as input; however, matting is still underconstrained
in the undefined area in the trimap. I created dataset and model which predict the trimap for different of kind of objects.
 
 
 
# Dependencies
- Python3.6+

# Tested on

- Windows 11, Python 3.6.1, Tensorflow 1.10.0, CUDA 10.0

# Model results:


![alt text](https://github.com/dsabarinathan/TrimapExtraction/blob/main/Image_Mat.jpg)


# References:

[Context-Aware Image Matting for Simultaneous Foreground and Alpha
Estimation](https://arxiv.org/abs/1909.09725)

[Densely Connected Convolutional Networks
](https://arxiv.org/abs/1608.06993)
