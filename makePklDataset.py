import pickle
import SimpleITK as sitk
import numpy as np
import glob
from natsort import natsorted
import os

def pksave(img, label, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump((img, label), f)

def nii2arr(nii_img):
    return sitk.GetArrayFromImage(sitk.ReadImage(nii_img))

def center(arr):
    c = np.sort(np.nonzero(arr))[:,[0,-1]]
    return np.mean(c, axis=-1).astype('int16')

def minmax(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def cropByCenter(image,center,final_shape=(160,192,160)):
    c = center
    crop = np.array([s // 2 for s in final_shape])
    # 0 axis
    cropmin, cropmax = c[0] - crop[0], c[0] + crop[0]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[0]
    if cropmax > image.shape[0]:
        cropmax = image.shape[0]
        cropmin = image.shape[0] - final_shape[0]
    image = image[cropmin:cropmax, :, :]
    # 1 axis
    cropmin, cropmax = c[1] - crop[1], c[1] + crop[1]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[1]
    if cropmax > image.shape[1]:
        cropmax = image.shape[1]
        cropmin = image.shape[1] - final_shape[1]
    image = image[:, cropmin:cropmax, :]

    # 2 axis
    cropmin, cropmax = c[2] - crop[2], c[2] + crop[2]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[2]
    if cropmax > image.shape[2]:
        cropmax = image.shape[2]
        cropmin = image.shape[2] - final_shape[2]
    image = image[:, :, cropmin:cropmax]
    return image

path_to_LPBA='/data/LPBA40/' # the path of the original dataset
img_niis = natsorted(glob.glob(path_to_LPBA+'*/*/*skullstripped.img.gz'))
label_niis = natsorted(glob.glob(path_to_LPBA+'*/*/*label.img.gz'))
print(img_niis, label_niis)

save_path = 'LPBA_data/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i, nii in enumerate(zip(img_niis, label_niis)):
    print(nii)
    img_nii, label_nii = nii
    img, label = nii2arr(img_nii), nii2arr(label_nii)
    print(img.shape, label.shape)
    
    # crop by center
    c = center(img)
    img = cropByCenter(img, c)
    label = cropByCenter(label, c)

    #norm
    img = minmax(img).astype('float32')
    label = label.astype('uint16')
    print(img.shape,np.unique(img),label.dtype, label.shape,np.unique(label),label.dtype)
    print(save_path+'subject_%02d.pkl'%(i+1))
    pksave(img,label, save_path=save_path+'subject_%02d.pkl'%(i+1))
