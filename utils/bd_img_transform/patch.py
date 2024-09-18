# the callable object for BadNets attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import numpy as np
import torch
from typing import Optional, Union
from torchvision.transforms import Resize, ToTensor, ToPILImage

class AddPatchTrigger(object):
    '''
    assume init use HWC format
    but in add_trigger, you can input tensor/array , one/batch
    '''
    def __init__(self, trigger_loc, trigger_ptn):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if isinstance(img, np.ndarray):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[m, n, :] = self.trigger_ptn[i]  # add trigger
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n, :] = self.trigger_ptn[i]  # add trigger
        elif isinstance(img, torch.Tensor):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n] = self.trigger_ptn[i]
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, :, m, n] = self.trigger_ptn[i]
        return img

class AddMaskPatchTrigger(object):
    def __init__(self,
                 trigger_array : Union[np.ndarray, torch.Tensor],
                 ):
        self.trigger_array = trigger_array

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        # print("------------------img.shape------------------------", img.shape)
        # print("------------------self.trigger_array.shape------------------------", self.trigger_array.shape)

        # img=img.reshape((64, 64, 3))  #reshape会改变数据排列 
        # self.trigger_array=self.trigger_array.reshape((3,64,64))  #reshape会改变数据排列 
        # if img.shape == (3, 64, 64):
        #     img=np.transpose(img, (1,2,0)) 
        # if self.trigger_array.shape == (64,64,3):
        #     self.trigger_array=np.transpose(self.trigger_array, (2,0,1))   #只改变通道排列
        # print("-------------self.trigger_array.shape----------img.shape-------------", self.trigger_array.shape, img.shape)

        if len(img.shape)==2:
            tmp=self.trigger_array.shape
            # print(tmp)
            trigger_array =  np.dsplit(self.trigger_array, 3)[0].reshape((tmp[0], tmp[1]))
            # print("------------------trigger_array.shape------------------------", trigger_array.shape)
            return img * (trigger_array == 0) + trigger_array * (trigger_array > 0)
        else:
            return img * (self.trigger_array == 0) + self.trigger_array * (self.trigger_array > 0)

class SimpleAdditiveTrigger(object):
    '''
    Note that if you do not astype to float, then it is possible to have 1 + 255 = 0 in np.uint8 !
    '''
    def __init__(self,
                 trigger_array : np.ndarray,
                 ):
        self.trigger_array = trigger_array.astype(np.float)

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return np.clip(img.astype(np.float) + self.trigger_array, 0, 255).astype(np.uint8)

import matplotlib.pyplot as plt
def test_Simple():
    a = SimpleAdditiveTrigger(np.load('../../resource/lowFrequency/cifar10_densenet161_0_255.npy'))
    plt.imshow(a(np.ones((32,32,3)) + 255/2))
    plt.show()
