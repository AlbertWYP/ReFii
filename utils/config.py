import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = ["zhaolian_train"]
    datasets_test = ["adm_res_abs_ddim20s"]
    mode = "binary"
    class_bal = False
    batch_size = 128#128
    loadSize = 256
    cropSize = 224
    epoch = "latest"
    num_workers = 20
    serial_batches = False
    isTrain = True

    # data augmentation
    rz_interp = ["bilinear"]
    blur_prob = 0.5
    blur_sig = [2]
    jpg_prob = 0#0
    jpg_method = ["cv2"]
    jpg_qual = [30]#75
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True
    aug_flip = True
    aug_norm = True

    ####### train setting ######
    warmup = False
    warmup_epoch = 3
    earlystop = True #true
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 400
    beta1 = 0.9
    lr = 0.0001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True

    ####### unetresnet setting ######
    class_cond=False
    image_size=64
    clip_denoised=True
    real_step=0
    use_ddim=False

    images_dir="/data2/wangzd/dataset/DiffusionForensics/images"
    recons_dir="/data2/wangzd/dataset/DiffusionForensics/recons"
    dire_dir="/data2/wangzd/dataset/DiffusionForensics/dire"
    clip_denoised=True
    num_samples=-1
    #batch_size=16,
    use_ddim=False
    model_path=""
    real_step=0
    continue_reverse=False
    has_subfolder=False

    #ddim
    num_channels=128,
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=-1
    attention_resolutions="16,8"
    channel_mult=""
    dropout=0.0
    #class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=False
    use_fp16=False
    use_new_attention_order=False
    
    learn_sigma=False,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,

    # paths information
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(root_dir, "data")
    exp_root = os.path.join(root_dir, "data", "exp")
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list):
    assert len(arg_list) % 2 == 0, f"Override list has odd length: {arg_list}; it must be a list of pairs"
    return dict(zip(arg_list[::2], arg_list[1::2]))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


CONFIGCLASS = Type[DefaultConfigs]

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default=[5], type=int, nargs="+")
parser.add_argument("--exp_name", default="", type=str)
parser.add_argument("--ckpt", default="model_epoch_latest.pth", type=str)
parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
args = parser.parse_args()

if os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    from config import cfg

    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

if args.opts:
    opts = args_list2dict(args.opts)
    for k, v in opts.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unrecognized option: {k}")
        original_type = type(getattr(cfg, k))
        if original_type == bool:
            setattr(cfg, k, str2bool(v))
        elif original_type in (list, tuple, set):
            setattr(cfg, k, str2list(v, type(getattr(cfg, k)[0])))
        else:
            setattr(cfg, k, original_type(v))

cfg.gpus: list = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])
cfg.exp_name = args.exp_name
cfg.ckpt_path: str = os.path.join(cfg.ckpt_dir, args.ckpt)

if isinstance(cfg.datasets, str):
    cfg.datasets = cfg.datasets.split(",")
