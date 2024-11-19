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
在训练之应该将训练`real`和`fake`图像放置到`data/train`文件夹。例如，将`LSUN-Bedroom`的`real`图像放置到`data/train/lsun_adm/0_real`目录下，将`ADM-LSUN-Bedroom`的虚假图像放置到`data/train/lsun_adm/1_fake`目录下。我们需要对验证集和测试集执行同样的操作，只需将`data/train`修改为`data/val`和`data/test`。此外，如果您想修改模型训练数据，请在`train.sh`修改`DATASETS`以及`DATASETS_TEST`路径，然后，执行以下命令训练ReFii模型：
```bash
sh train.sh
```
## Evaluation
在训练完成后，我们为您保存了最佳模型，其路径为`/data/exp/your_best_model`。如果您想用这个模型进行测试，那您需要在`test.sh`中修改`CKPT`路径为您的最优模型路径，然后通过运行下面的程序即可进行测试。
```bash
sh test.sh
```
## Intra-domain discrimination
例如，在数据集`DiFF`上进行`T2I`域内鉴别，我们需要按照如下步骤进行：首先，我们需要按照创建`Training`中的方法创建`DiFF_T2I`子数据集，并且将`train.sh`中的`DATASETS`以及`DATASETS_TEST`均修改为`DiFF_T2I`，在训练完成后，我们将在路径`data/exp/DiFF_T2I/ckpt/model_epoch_best.pth`获得其最优模型；在测试阶段，我们需要按照`Evaluation`的方法进行域内测试，其中`test.sh`中`CKPT`路径为`data/exp/DiFF_T2I/ckpt/model_epoch_best.pth`，`DATASETS_TEST`为`DiFF_T2I`。
## Cross-domain discrimination
例如，在数据集`DiFF`上进行`T2I`训练`I2I`的跨域鉴别，我们需要按照如下步骤进行：训练过程同`Intra-domain discrimination`；在测试阶段，我们需要将`test.sh`中`DATASETS_TEST`修改为`DiFF_I2I`。
## Robustness experiment
鲁棒性测试我们主要针对`JPEG压缩`和`高斯模糊`进行实验。在`/utils/config.py`中参数`jpg_qual`和`jpg_prob`分别控制JPEG压缩程度以及JPEG压缩概率，`blur_sig`和`blur_prob`分别控制高斯模糊强度以及高斯模糊概率，我们只需要修改这几个参数，并进行测试步骤即可完成鲁棒性实验。
