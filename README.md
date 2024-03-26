# Recursive Deformable Pyramid Network for Unsupervised Medical Image Registration (TMI2024)

By Haiqiao Wang, Dong Ni, Yi Wang.

Paper link: [[TMI]](https://ieeexplore.ieee.org/document/10423043)

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-11.3-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
## Description
An unsupervised brain MR deformable registration method that achieves precise alignment through a pure convolutional pyramid structure and a semantics-infused progressive recursive inter-level looping strategy for modeling complex deformations, even without pre-alignment of brain MR images.

![图片1](https://github.com/ZAX130/RDP/assets/43944700/66c3058f-7d9c-499c-8017-40c62240f4d7)


## Dataset
The official access addresses of the public data sets are as follows：

LPBA [[link]](https://resource.loni.usc.edu/resources/atlases-downloads/) 

Mindboggle [[link]](https://osf.io/yhkde/)

IXI [[link]](https://surfer.nmr.mgh.harvard.edu/pub/data/) [[freesurfer link]](https://surfer.nmr.mgh.harvard.edu/pub/data/ixi/)

Note that we use the processed IXI dataset provided by freesurfer.

## Instructions
For convenience, we are sharing the preprocessed [LPBA](https://drive.usercontent.google.com/download?id=1mFzZDn2qPAiP1ByGZ7EbsvEmm6vrS5WO&export=download&authuser=0) dataset used in our experiments. Once uncompressed, simply modify the "LPBA_path" in `train.py` to the path name of the extracted data. Next, you can execute `train.py` to train the network, and after training, you can run `infer.py` to test the network performance.

## Citation
If you use the code in your research, please cite:
```
@ARTICLE{10423043,
  author={Wang, Haiqiao and Ni, Dong and Wang, Yi},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Recursive Deformable Pyramid Network for Unsupervised Medical Image Registration}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Deformation;Decoding;Feature extraction;Deformable models;Training;Image resolution;Image registration;Deformable image registration;convolutional neural networks;brain MRI},
  doi={10.1109/TMI.2024.3362968}}
```
The overall framework and some network components of the code are heavily based on [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph). We are very grateful for their contributions.

The file `makePklDataset.py` shows how to make a pkl dataset from the original LPBA dataset. If you have any other questions about the .pkl format, please refer to the github page of [[TransMorph_on_IXI]](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md). 

## Baseline Methods
Several PyTorch implementations of some baseline methods can be found at [[SmileCode]](https://github.com/ZAX130/SmileCode/tree/main).
