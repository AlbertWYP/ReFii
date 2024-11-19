
import torch
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th
from tqdm import tqdm
import torch.distributed as dist

import sys
sys.path.append("/data1/yipeng_wei/DIRE_change/guided-diffusion/") # 将module1所在的文件夹路径放入sys.path中
from guided_diffusion import dist_util


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs

class UNetResNet(nn.Module):  
    def __init__(self, unet1,resnet1,diffusion1):  
        super(UNetResNet, self).__init__()
        self.unet = unet1
        self.resnet = resnet1
        self.output=nn.Linear(in_features=1000, out_features=1, bias=True)
        self.diffusion = diffusion1
    def compute_dire(self,x,*args):#x是一个图片
        ##要修改一下下面这些引用
        img = x
        batch_size=1
        img = img.to(dist_util.dev())
        unet_kwargs = {}
        ###超参数设置
        class_cond=False
        image_size=64
        clip_denoised=True
        real_step=0
        use_ddim=False
        NUM_CLASSES=1000
        ###
        
        if class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev())
            unet_kwargs["y"] = classes
        reverse_fn = self.diffusion.ddim_reverse_sample_loop
        img = reshape_image(img, image_size)

        latent = reverse_fn(
            self.unet,
            (batch_size, 3, image_size, image_size),
            noise=img,
            clip_denoised=clip_denoised,
            model_kwargs=unet_kwargs,
            real_step=real_step,
        )
        sample_fn = self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop
        recons = sample_fn(
            self.unet,
            (batch_size, 3, image_size, image_size),
            noise=latent,
            clip_denoised=clip_denoised,
            model_kwargs=unet_kwargs,
            #real_step=real_step,
        )

        dire = th.abs(img - recons)
        dire = (dire * 255.0).clamp(0, 255) / 255.0
        #dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
        out = dire
        
        #for data in tqdm(dire, dynamic_ncols=True):

        # dire_save_dir = "/data1/yipeng_wei/DIRE_change/pic/dire"  
        # recons_save_dir = "/data1/yipeng_wei/DIRE_change/pic/recons"  
        
        # # 确保目录存在，如果不存在则创建  
        # os.makedirs(dire_save_dir, exist_ok=True)  
        # os.makedirs(recons_save_dir, exist_ok=True)  
    
        # fn_save = f"1.png"  
        # dire = th.abs(img - recons)
        # recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # recons = recons.permute(0, 2, 3, 1)
        # recons = recons.contiguous()

        # img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # img = img.permute(0, 2, 3, 1)
        # img = img.contiguous()

        # dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
        # dire = dire.permute(0, 2, 3, 1)
        # dire = dire.contiguous()
        # recons = recons.cpu().numpy()  
            
        # # 保存dire图像  
        # cv2.imwrite(  
        #     os.path.join(dire_save_dir, fn_save),  
        #     cv2.cvtColor(dire.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)  
        # )  
        
        # # 保存recons图像  
        # cv2.imwrite(  
        #     os.path.join(recons_save_dir, fn_save),  
        #     cv2.cvtColor(recons.astype(np.uint8), cv2.COLOR_RGB2BGR)  
        # )

        return out.float() 


    def forward(self, x, *args):
        dire=self.compute_dire(x,args)
        # print("dire")
        # print(dire.size())
        #转换成numpy会使梯度消失
        # image_array = x.numpy().astype(np.uint8)  
        # # 如果图像是RGB格式，转换为BGR格式以便保存  
        # image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  
        x=self.resnet(dire)
        output=self.output(x)
        # print("网络内的输出")
        # print(output.size()) 
        return output
    


# import torch as th  
# import torch.nn as nn  
# # 假设你已经有了扩散模型（diffusion1）、UNet（unet1）和ResNet（resnet1）的实现  
  
# class UNetResNet(nn.Module):  
#     def __init__(self, unet1, resnet1, diffusion1):  
#         super(UNetResNet, self).__init__()  
#         self.unet = unet1  
#         self.resnet = resnet1  
#         self.diffusion = diffusion1  
  
#     def compute_dire(self, x):  
#         # 确保x是一个Tensor，并且已经移动到了正确的设备上  
#         x = x.to(self.device)  
          
#         # 超参数设置（可以根据需要进行调整）  
#         class_cond = False  
#         image_size = 64  
#         clip_denoised = True  
#         real_step = 0  
#         use_ddim = False  
#         NUM_CLASSES = 1000  
          
#         # 如果需要类别条件，确保classes也是一个Tensor  
#         if class_cond:  
#             classes = th.randint(low=0, high=NUM_CLASSES, size=(x.shape[0],), device=self.device)  
#             unet_kwargs = {"y": classes}  
#         else:  
#             unet_kwargs = {}  
          
#         # 使用扩散模型的逆向采样函数  
#         reverse_fn = self.diffusion.ddim_reverse_sample_loop  
#         imgs = reshape_image(x, image_size)  # 假设reshape_image是一个可导的操作或函数  
          
#         latent = reverse_fn(  
#             self.unet,  
#             (x.shape[0], 3, image_size, image_size),  
#             noise=imgs,  
#             clip_denoised=clip_denoised,  
#             unet_kwargs=unet_kwargs,  
#             real_step=real_step,  
#         )  
          
#         # 使用扩散模型的采样函数  
#         sample_fn = self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop  
#         recons = sample_fn(  
#             self.unet,  
#             (x.shape[0], 3, image_size, image_size),  
#             noise=latent,  
#             clip_denoised=clip_denoised,  
#             unet_kwargs=unet_kwargs,  
#             real_step=real_step,  
#         )  
          
#         # 计算差异  
#         dire = imgs - recons  
          
#         # 将差异缩放到0-255范围并转换为uint8（这里假设不需要梯度，因为通常不会反向传播到差异图像）  
#         dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)  
          
#         # 注意：如果dire用于显示或保存，而不是用于计算梯度，那么将其转换为uint8是可以的。  
#         # 但是，如果dire需要进一步参与计算并且需要梯度，那么应该保持为float类型。  
          
#         # 返回差异Tensor，保持为float类型以便可能后续使用  
#         return dire.float()  
  
#     def forward(self, x):  
#         # 确保模型在正确的设备上  
#         self.to(x.device)  
#         self.device = x.device  # 更新设备信息，假设在初始化时未设置  
          
#         # 计算差异  
#         dire = self.compute_dire(x)  
          
#         # 将差异图像传递给ResNet模型（确保模型接受Tensor作为输入）  
#         resnet_output = self.resnet(dire)  
          
#         return resnet_output