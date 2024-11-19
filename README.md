# ReFii
## Requirements
```bash
conda create -n refii python=3.9
conda activate refii
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
## Datasets
我们用到了[DiFF](https://github.com/xaCheng1996/DiFF)以及[DiffusionForensics](https://pan.baidu.com/share/init?surl=Rdzc7l8P0RrJft0cW0a4Gg)数据集，数据集下载链接已经给出，其中`DiffusionForensics`数据集密码为`dire`。

## Training
在训练之应该将训练real和fake图像放置到data/train文件夹。例如，将LSUN-Bedroom的real图像放置到`data/train/lsun_adm/0_real`目录下，将ADM-LSUN-Bedroom的虚假图像放置到`data/train/lsun_adm/1_fake`目录下。我们需要对验证集和测试集执行同样的操作，只需将`data/train`修改为`data/val`和`data/test`。然后，执行以下命令训练ReFii模型：
