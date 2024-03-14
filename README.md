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
LPBA [[link]](https://resource.loni.usc.edu/resources/atlases-downloads/)

Mindboggle [[link]](https://osf.io/yhkde/)

IXI [[link]](https://surfer.nmr.mgh.harvard.edu/pub/data/) [[freesurfer link]](https://surfer.nmr.mgh.harvard.edu/pub/data/ixi/)

Note that we use the processed IXI dataset provided by freesurfer.

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
The overall framework and some network components of the code are heavily based on [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph). We are very grateful for their contributions. If you have any questions about the .pkl format, please refer to the github page of TransMorph. The file makePklDataset.py shows how to make a pkl dataset from the original LPBA dataset.
