import torch
import torch.nn as nn
import torch.nn.functional as F
from pydoc import locate
from transformers import BertTokenizer
import sys
from torchvision import transforms
from einops import rearrange, reduce, repeat
from PIL import Image
import hydra
from einops import rearrange, reduce, repeat
import types

LILT_ABSOLUTE = "/home/raiymbek/vlm/project2/lilt/LilT"
LILT_RELATIVE = "../../../LilT"
CHECKPOINT_DIR = "/home/raiymbek/vlm/project2/lilt/LilT/results"
sys.path.insert(0, LILT_ABSOLUTE)

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import build

def channel_check(x):
    if x.shape[0] == 3:
        return x
    #print("got here")
    return repeat(x, "c h w -> (c rep) h w", rep = 3)


def encode_text(self, text):
    return self.eval_text_forward(text)

def encode_image(self, image):
    return self.eval_image_forward(image)


def load_lilt(config, checkpoint, device="cpu"):
    #config = "dinov2-clip-wds-combined.yaml"
    output_dir = "./results"
    with hydra.initialize(config_path=f"{LILT_RELATIVE}/configs-v2"):
        config = hydra.compose(config_name=config)

    tokenizer_de = build.tokenizer(config)
    tokenizer = lambda text: tokenizer_de(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt"
    )

    model_class = locate(config.model_config.import_path)
    model = model_class(
        config=config, text_encoder=config.text_encoder, tokenizer=tokenizer
    )
    model.encode_text = types.MethodType(encode_text, model)
    model.encode_image = types.MethodType(encode_image, model)
    
    #checkpoint = torch.load("../LilT/results/checkpoint_14_gradaccum22.pth", map_location="cpu")
    checkpoint = torch.load(f"{CHECKPOINT_DIR}/{checkpoint}", map_location="cpu")
    state_dict = checkpoint["model"]
    
    required_keys = model.state_dict().keys()
    state_dict = {k: v for k, v in state_dict.items() if k in required_keys}
    msg = model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            channel_check,
        ]
    )
    return model, transform, tokenizer
    

    