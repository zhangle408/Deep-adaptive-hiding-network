# Deep-adaptive-hiding-network
Deep adaptive hiding network for image hiding using attentive frequency extraction and gradual depth extraction
## Requirements
This code was developed and tested with Python3.6, Pytorch 1.5 and CUDA 10.2 on Ubuntu 18.04.5.

## Train DAH-Net on ImageNet datasets
You are able to run the provided demo code.

1. Prepare the ImageNet datasets and visualization dataset.

2. Change the data path on lines 210-214 of train_dah.py.

   (Images for training exist in traindir and valdir, and images for visualization exist in coverdir and secretdir ).

3. ''' sh ./scripts/train_dah.sh '''

## Citing
If you found our research helpful or influential please consider citing


### BibTeX
@article{zhang2023deep,
  title={Deep adaptive hiding network for image hiding using attentive frequency extraction and gradual depth extraction},
  author={Zhang, Le and Lu, Yao and Li, Jinxing and Chen, Fanglin and Lu, Guangming and Zhang, David},
  journal={Neural Computing and Applications},
  pages={1--19},
  year={2023},
  publisher={Springer}
}
