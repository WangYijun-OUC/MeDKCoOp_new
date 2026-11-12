import os.path as osp
import os
import sys
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
# from . import models_mae, models_mae_vitae
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import numpy as np
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from MedSAM.segment_anything import sam_model_registry
from clip.pmcclip import ModifiedResNet

from transformers import AutoTokenizer, AutoModel

# Directory where the files should be located
directory = os.environ.get("PMCCLIP_DIR", "./weights/PubMed")
BERT_DIR = os.environ.get(
    "PMCCLIP_BERT_DIR", "./weights/BiomedCLIP/BiomedBERT"
)

# File URLs
files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}

# Function to download a file
def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            # Use tqdm to show the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))  # Update progress bar by the chunk size
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")

class TextEncoder(nn.Module):
    def __init__(self, pmcclip_model):
        super().__init__()
        self.model = pmcclip_model
        self.dtype = torch.float32
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer

    def forward(self, prompts,tokenized_prompts):

        output = self.text_encoder(inputs_embeds=prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
        pooler_output = output.pooler_output
        text_feature = pooler_output @ self.text_projection_layer

        return text_feature


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        n_cls = len(classnames)         # 类数量 base/new类减半
        n_ctx = cfg.TRAINER.DCPL.N_CTX  # 上下文词数量 2
        ctx_init = cfg.TRAINER.DCPL.CTX_INIT    # a photo of a
        
        dtype = torch.float32
        ctx_dim = 768
        clip_imsize = 224
        
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_DIR, local_files_only=True)
        assert cfg.TRAINER.DCPL.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.DCPL.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = self.tokenizer(ctx_init, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']
            with torch.no_grad():
                embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(prompt.cuda()).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('DCPL design: Domain Controlled')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of DCPL context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers
        # self.meta_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(512, 512 // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512 // 16, ctx_dim))
        # ]))

        print("CLIP CTX_DIM=", ctx_dim)     # 512
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(256, 256 // 16)),     # W1=[256, 16], b1=16, features*W1+b1 -> [4, 16]
            ("relu", nn.ReLU(inplace=True)),            # 激活函数
            ("linear2", nn.Linear(256 // 16, ctx_dim))  # W2=[16, 512], b2=512, features*X2+b2 -> [4, 512]
        ]))

        if cfg.TRAINER.DCPL.PREC == "fp16":
            self.meta_net.half()
        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]        # ['glioma', 'meningioma'] for base
        name_lens = [len(self.tokenizer(name, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']) \
                     for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # ['a photo of a glioma.', 'a photo of a meningioma.']

        tokenized_prompts = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        with torch.no_grad():
            embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(tokenized_prompts['input_ids'].cuda()).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS    [2, 1, 512]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS      [2, 74, 512]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (4, during training) or n_cls (2, during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim=512)   [2, 1, 512]
                ctx,  # (dim0, n_ctx, dim=512)  [2, 2, 512]
                suffix,  # (dim0, *, dim=512)   [2, 74, 512]
            ],
            dim=1,
        )

        # 得到prompts大小为 [2, 77, 512]

        return prompts  

    def forward(self,maefeatures):
        ctx = self.ctx # (n_ctx, ctx_dim)   [2, 512]
        
        bias = self.meta_net(maefeatures)  # (batch, ctx_dim)   [4, 512]
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)         [4, 1, 512]
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)           [1, 2, 512]
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)     [4, 2, 512]
        prefix = self.token_prefix  # [2, 1, 512]
        suffix = self.token_suffix  # [2, 74, 512]
        prompts = []
        for ctx_shifted_i in ctx_shifted:   
            # 每个batch ctx_shifted_i [2, 512]
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)   # [2, 2, 512]   第0维扩展到n_cls位，复制操作
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)           # (n_cls, n_tkn, ctx_dim)   [2, 77, 512]
            prompts.append(pts_i)                                           # 四次 [2, 77, 512]
        prompts = torch.stack(prompts)                                      # [4, 2, 77, 512]

        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # prefix = self.token_prefix
        # suffix = self.token_suffix
        # prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        # visual_deep_prompts = []
        # for index, layer in enumerate(self.compound_prompt_projections):
        #     visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts   # pass here original, as for visual 768 is required

class PMCCLIP(nn.Module):
    def __init__(self,image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_DIR, local_files_only=True)
    def forward(self,image,text):
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        text_feature = self.text_encoder(input_ids)
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']

        return image_feature, text_feature

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, pmcclip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = pmcclip_model.image_encoder
        self.text_encoder = TextEncoder(pmcclip_model)
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        
        # print('loading remote MAE model')
        # self.remote_model = models_mae.__dict__['mae_vit_base_patch16'](norm_pix_loss=False)
        # model_path = '/code/multimodal-prompt-learning/premodels/vit-b-checkpoint-1599.pth'
        # self.remote_model.load_state_dict(torch.load(model_path)['model'])

        # self.remote_model = models_mae_vitae.__dict__['mae_vitae_base_patch16'](norm_pix_loss=False)
        # model_path = '/code/multimodal-prompt-learning/premodels/vitae-b-checkpoint-1599-transform-no-average.pth'
        # self.remote_model.load_state_dict(torch.load(model_path))

        print('loading medical model')
        MedSAM_CKPT_PATH = "./weights/sam_vit_b_01ec64.pth"
        self.medical_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)

        # self.visual_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(512, 512 // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512 // 16, 512))
        # ]))
        self.visual_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(256, 256 // 16)),     # W1=[256, 16], b1=16, features*W1+b1 -> [4, 16]
            ("relu", nn.ReLU(inplace=True)),            # 激活函数
            ("linear2", nn.Linear(256 // 16, 768))      # W2=[16, 512], b2=512, features*X2+b2 -> [4, 512]
        ]))

        # self.p1 = nn.Parameter(torch.tensor([1.0]))
        # self.p2 = nn.Parameter(torch.tensor([0.0]))
        if cfg.TRAINER.DCPL.PREC == "fp16":
            self.visual_net.half()
            # self.p1.half()
            # self.p2.half()
    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale       # tensor(100., device='cuda:7')
        
        with torch.no_grad():
            # remote_mae_feature = self.remote_model(image)
            # remote_mae_feature = remote_mae_feature.mean(dim=1).type(self.dtype)
            # print(image.shape)  4*3*224*224
            # print(remote_mae_feature.shape) 4*196*512
            # print(image_features.shape) 4*196*512  

            medical_feature = self.medical_model.image_encoder(nn.functional.interpolate(image, (1024, 1024)))
            mb,mc,mh,mw = medical_feature.shape
            medical_feature = medical_feature.view(mb,mc,mh*mw)
            remote_mae_feature = medical_feature.mean(dim=-1).type(self.dtype)

        # print(image.shape)                  # [4, 3, 224, 224]
        # print(medical_feature.shape)        # [4, 256, 4096]
        # print(remote_mae_feature.shape)     # [4, 256]
        # print(remote_mae_feature.type(self.dtype))  # 4*256 tensor device=cuda dtype=float16
        # sys.exit()
        prompts = self.prompt_learner(remote_mae_feature.type(self.dtype))
        # text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features, dict):
            image_features = image_features['image_features']

        mean = torch.mean(image_features)
        # var = torch.var(image_features)
        # std = torch.sqrt(var)
        # noise = torch.normal(mean.cpu(), std.cpu(), size=image_features.shape)
        # noise = torch.tensor(np.random.normal(mean.cpu().detach().numpy(), std.cpu().detach().numpy(), size=image_features.shape)).cuda().type(self.dtype)
        noise = torch.randn(image_features.shape).cuda().type(self.dtype)*mean      # [4, 512]
        image_features_bias = self.visual_net(remote_mae_feature.type(self.dtype))  # [4, 512]
        image_features = image_features + image_features_bias +noise                # [4, 512]
        # image_features = image_features + image_features_bias

        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # [4, 512]
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = []
        # tokenized_prompts [2, 77]
        # pts_i [2, 77, 512]
        # deep_compound_prompts_text = List 8 * Parameter[2, 512]
        for pts_i, imf_i in zip(prompts, image_features.type(self.dtype)):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()   
            logits.append(l_i)  # [2] 当前样本与两个类别的相似度
        logits = torch.stack(logits)    # [4, 2]
        # logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class DCPL_PMCCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DCPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        print("Using Model DCPL...")
        # sys.exit()
        
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth')))

        # Load Text Encoder
        text_encoder = AutoModel.from_pretrained(BERT_DIR, local_files_only=True)
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth')))

        # Load Text Proj Layer

        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'))
        text_projection_layer = nn.Parameter(text_projection_layer)

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_encoder = image_encoder.to(device).eval()
        text_encoder = text_encoder.to(device).eval()
        text_projection_layer = text_projection_layer.to(device)

        pmcclip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()

        if cfg.TRAINER.DCPL.PREC == "fp32" or cfg.TRAINER.DCPL.PREC == "amp":
            # CLIP's default precision is fp16
            pmcclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                # if "VPT" in name or 'visual_net ' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            if "visual_net" in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DCPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # device_ids = [4]
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.DCPL.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
