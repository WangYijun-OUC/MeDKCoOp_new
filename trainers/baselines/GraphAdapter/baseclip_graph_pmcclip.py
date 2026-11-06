import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import os

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.model import convert_weights

# from .coop import load_clip_to_cpu
from trainers.prompt_templates import CUSTOM_TEMPLATES
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.utils import load_pretrained_weights, load_checkpoint
import math
from dassl.metrics import compute_accuracy
from clip.pmcclip import ModifiedResNet

from transformers import AutoTokenizer, AutoModel

# CUSTOM_TEMPLATES = {
#     "OxfordPets": "a photo of a {}, a type of pet.",
#     "OxfordFlowers": "a photo of a {}, a type of flower.",
#     "FGVCAircraft": "a photo of a {}, a type of aircraft.",
#     "DescribableTextures": "{} texture.",
#     "EuroSAT": "a centered satellite photo of {}.",
#     "StanfordCars": "a photo of a {}.",
#     "Food101": "a photo of {}, a type of food.",
#     "SUN397": "a photo of a {}.",
#     "Caltech101": "a photo of a {}.",
#     "UCF101": "a photo of a person doing {}.",
#     "ImageNet": "a photo of a {}.",
#     "ImageNetSketch": "a photo of a {}.",
#     "ImageNetV2": "a photo of a {}.",
#     "ImageNetA": "a photo of a {}.",
#     "ImageNetR": "a photo of a {}.",
#     "BTMRI": "a photo of a {}.",
#     "CHMNIST": "a photo of a {}.",
#     "CHMNISTv2": "a photo of a {}.",
#     "CCBTM": "a photo of a {}.",
#     "CCBTMv2": "a photo of a {}.",
# }

# Directory where the files should be located
directory = os.environ.get("PMCCLIP_DIR", "/data/wangyijun/model/PubMed")
BERT_DIR = os.environ.get(
    "PMCCLIP_BERT_DIR", "/data/wangyijun/model/BiomedCLIP/BiomedBERT"
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



def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
	# A = A + I    A: (bs, num_nodes, num_nodes
    # Degree
    d = A.sum(-1) # (bs, num_nodes) #[1000, m+1]
    if symmetric:
		# D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
		# D=D^-1
        d = torch.pow(d,-1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D =torch.diag(d)
            norm_A = D.mm(A)

    return norm_A

def cal_similarity(x, p=2, dim=1):
    '''
    x: (n,K)
    return: (n,n)
    '''
    x = F.normalize(x, p=p, dim=dim)
    return torch.mm(x, x.transpose(0, 1))

def cal_edge_emb(x, p=2, dim=1):   # v1_graph---taking the similairty by 
    ''' 
    x: (n,K)   [m+1, 1000, 1024]
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)    #[m+1, 1000, 1024], [100, 1024, 101]
    x_c = x
    x = x.transpose(1, 2)  #[1000, m+1, 1024]  [100, 101, 1024]
    x_r = x  # (K, n, 1) #[1000, m+1, 1024]
    # x_c = torch.transpose(x, 1, 2)  # (K, 1, n) #[1000, 1024, m+1]
    # A = torch.bmm(x_r, x_c).permute(1,2,0)  # (n, n, K) 
    A = torch.bmm(x_r, x_c)     # [1000, m+1, m+1]

    # A = A.view(A.size(0) * A.size(1), A.size(2))  # (n^2, K)
    # print(A.size())
    return A


class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True, dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device=device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs  # 是否使用稀疏输入
        self.hidden_dim = hidden_dim
        self.bias = bias
        # self.hidden_dim = 512
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))   # 可学习的权重矩阵 [hidden_dim, hidden_dim]   float32
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))       # [class_num, hidden_dim]
           
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))     # 初始化权重 gcn_weights
        self.gcn_weights.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.gcn_bias.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        # feat [batch, feature_dim, node_num]       [2, 1024, 3]
        # adj 邻接矩阵 [batch, node_num, node_num]  [2, 3, 3]
        x = feat                        
        node_size = adj.size()[1]       # 3
        adj = torch.clip(adj, min=0.0)  # [2, 3, 3]
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to(self.device)    # 单位矩阵  [1, 3, 3]
        adj = adj + I      # [1000, m+1, m+1]   [2, 3, 3]
        adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)  #[1000, m+1, m+1]   [2, 3, 3] 拉普拉斯对边变换 Dtt Dvt 得到ε^tt
        x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)     # [m+1, 1000, 1024]         Ctt * Wtt   [2, 3, 1024]
        output = torch.matmul(adj, pre_sup)             # [1000, m+1, 1024]         ε^tt * Ctt * Wtt     [2, 3, 1024]

        if self.bias:
            output += self.gcn_bias.unsqueeze(1)
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]      # [2, 1024]




class GraphLearner(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model, base_text_features, base_img_features):
        super().__init__()
        self.device = torch.float32
        # self.alpha = cfg.TRAINER.COOP.RESIDUAL_SCALE
        self.alpha=0.1
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features)  #[1000, 1024]
        self.register_buffer("base_img_features", base_img_features)
        # self.alpha_it = cfg.TRAINER.GRAPHADAPTER.ALPHA
        self.alpha_it = 0.7
        self.beta_it = cfg.TRAINER.GRAPHADAPTER.BETA    # 0.3
        self.node_num = 1
        self.hidden_dim = 768
        self.GCN_tt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])   # 2
        self.GCN_it = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device, class_num=base_text_features.size()[0])

    def reset_parameters(self):
        for i in range(self.node_num):
            stdv = 1. / math.sqrt(self.graph_node[i].size(0))
            self.graph_node[i].data.uniform_(-stdv, stdv)

    def forward(self, img_feature):
        sigma=2.0

        with torch.no_grad():
            # 将基础特征分成四个子图
            node_cluster_t = self.base_text_features.view(1, self.base_text_features.size()[0]//1, 1, self.base_text_features.size()[1])    # [1, 2, 1, 1024]
            node_cluster_i = self.base_img_features.view(1, self.base_img_features.size()[0]//1, 1, self.base_img_features.size()[1])
           
        graph_o_t_all = []
        # pdb.set_trace()

        for index in range(1):
            # print("========index", index)
            with torch.no_grad():
                inputs_text = self.base_text_features.unsqueeze(dim=1)    #[2, 1, 1024]       zt
                # inputs_img = img_feature.unsqueeze(dim=1)
                node_cluster_tt =  node_cluster_t[:, :, index, :].repeat(inputs_text.size()[0], 1, 1)  # t->t [1,2,1024]->[2,2,1024]
                node_cluster_it =  node_cluster_i[:, :, index, :].repeat(inputs_text.size()[0], 1, 1)  # i->t [2,2,1024]
                feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1)      # [2, 3, 1024]
                feat_it = torch.cat([inputs_text, node_cluster_it], dim=1)      # [2, 3, 1024]
                feat_tt = feat_tt.transpose(1, 2).detach()          # [2, 1024, 3]
                feat_it = feat_it.transpose(1, 2).detach()          # [2, 1024, 3]
                edge_tt = cal_edge_emb(feat_tt).detach()    # εtt zt与文本子图Ct的边    [2, 3, 3]
                edge_it = cal_edge_emb(feat_it).detach()    # εvt zt与图像子图Cv的边    [2, 3, 3]
            graph_o_tt = self.GCN_tt(feat_tt, edge_tt)      # 获取adapted文本特征       [2, 1024]
            graph_o_it = self.GCN_it(feat_it, edge_it)      # 获取refined文本特征       [2, 1024]
            graph_o_t = (graph_o_tt)*self.alpha_it + (1-self.alpha_it)*graph_o_it   # 融合内模态和跨模态特征 = b*ztt + (1-b)*zvt    [2, 1024]
            graph_o_t_all.append(graph_o_t)
        graph_o_t = torch.stack(graph_o_t_all, dim=0).mean(dim=0)   # [2, 1024]
    
        return self.beta_it * self.base_text_features + (1-self.beta_it) * graph_o_t.squeeze(), img_feature # 最终特征 = a*zt + (1-a)*z't


def _get_base_text_features(cfg, classnames, pmcclip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    dtype = torch.float32     # float32
    if dtype == torch.float32:
        text_encoder = text_encoder.cuda()
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR, local_files_only=True)
    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = tokenizer([template.format(text) for template in TEMPLATES], padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']   # [1, 77]
            # x_tokenized = torch.cat([tokenizer(BIOMEDCOOP_TEMPLATES[name][i], padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']])    # [1, 256] int64
            x_tokenized_attn_masks = tokenizer([template.format(text) for template in TEMPLATES], padding='max_length', truncation=True, max_length=77, return_tensors='pt')['attention_mask']
            # tokens = tokens.to(device)
            # print("=============", tokens.dtype, clip_model.dtype)
              # tokenized prompts are indices
            # embeddings = pmcclip_model.text_encoder.embeddings.word_embeddings(tokens.cuda()).type(dtype)
            if dtype == torch.float32:
                text_embeddings.append(pmcclip_model.text_encoder(tokens.cuda(),x_tokenized_attn_masks.cuda()).pooler_output)  # not support float16 on cpu    [1, 1024]
            else:
                text_embeddings.append(pmcclip_model.text_encoder(tokens.cuda(),x_tokenized_attn_masks.cuda()).pooler_output)
    text_embeddings = torch.stack(text_embeddings).mean(1)  # [class_num, feature_dim]      [2, 1024]
    text_encoder = text_encoder.to(device)
    
    return text_embeddings.to(device)



def _get_base_image_features(cfg, classnames, pmcclip_model, img_encoder, train_loader_x):
    device = next(img_encoder.parameters()).device
    dtype = torch.float32     # float32
    if dtype == torch.float32:
        img_encoder = img_encoder.cuda()
    # text_encoder = text_encoder.cuda()
    # dataset = cfg.DATASET.NAME
    with torch.no_grad():
        img_feature = []
        labels = []
        # for epch in range(10):
        for batch_idx, batch in enumerate(train_loader_x):  # 一个存4张图，有8个这个，一共32
            image = batch["img"]
            label = batch["label"]
            image = image.cuda()
            label = label.cuda()
            image_features = img_encoder(image.type(dtype))     # [4, 1024]
            if isinstance(image_features, dict):
                image_features = image_features['image_features']
            img_feature.append(image_features)    
            labels.append(label)
        
        # # 合并所有特征和标签
        
        # img_feature_list = torch.cat(img_feature, dim=0)                    # [total_samples, feature_dim]  [80]*[4, 1024] = [320, 1024]
        # label_list = torch.cat(labels, dim=0)                               

        # sorted, indices = torch.sort(label_list)                            # [320]
        # print("+++++++++++++++++++len_label", len(sorted), sorted[-1])      # 320 tensor(1, device='cuda:0')

        # # 每个类别的样本数
        # label_len = len(sorted)//(sorted[-1]+1) # 320 / 1 + 1 = 160
        # print('=====label_len', label_len)      # tensor(160, device='cuda:0')

        # img_feature_list_all = torch.index_select(img_feature_list, 0, indices) # [320, 1024]   在第0维选择320个元素，即全选
        # b, c = img_feature_list_all.size()                                      # 320 1024
        # label_list = sorted.view(b//label_len, label_len)                       # 2， 160
        # img_feature_list_all = img_feature_list_all.view(b//label_len, label_len, -1).mean(dim=1)   # [2, 160, 1024] -> [2, 1024]
        # img_encoder = img_encoder.to(device)    # [class_num, feature_dim]
        img_feature_list = torch.cat(img_feature, dim=0)
        label_list = torch.cat(labels, dim=0)

        sorted_labels, indices = torch.sort(label_list)     # 10837     [10837]
        img_feature_sorted = torch.index_select(img_feature_list, 0, indices)

        # 获取每个类别的唯一值和样本数量
        unique_labels, counts = torch.unique(sorted_labels, return_counts=True)     # [1533, 3154, 3196, 2954]
        num_classes = unique_labels.size(0)

        # 根据实际样本数分割特征张量
        split_indices = counts.cumsum(dim=0)[:-1].tolist()      # [1533, 4687, 7883]
        features_split = torch.tensor_split(img_feature_sorted, split_indices, dim=0)       # [1533, 512] ... [2954, 512]

        # 计算每个类别的平均特征
        class_means = [feat.mean(dim=0) for feat in features_split]     
        img_feature_means = torch.stack(class_means, dim=0)  # 形状为[num_classes, feature_dim]
        img_encoder = img_encoder.to(device)
        pmcclip_model = pmcclip_model.to(device)
    return img_feature_means.to(device)  # [2, 512]


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
    def __init__(self, cfg, classnames, pmcclip_model, train_loader_x):
        super().__init__()
        self.image_encoder = pmcclip_model.image_encoder
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        text_encoder = TextEncoder(pmcclip_model)
        img_encoder = self.image_encoder
        base_text_features = _get_base_text_features(cfg, classnames, pmcclip_model, text_encoder)
        base_img_features = _get_base_image_features(cfg, classnames, pmcclip_model, img_encoder, train_loader_x)

        self.graph_learner = GraphLearner(cfg, classnames, pmcclip_model, base_text_features, base_img_features)

    def forward(self, image):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except:
            image_features = self.image_encoder(image.float())
        if isinstance(image_features, dict):
            image_features = image_features['image_features']
        text_features, image_features = self.graph_learner(image_features)      # [2, 1024], [4, 1024]
       

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class GraphCLIP_PMCCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
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

        if cfg.TRAINER.COOP.PREC == "fp16" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            pmcclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model, self.train_loader_x)
        # for key, value in self.model.named_parameters():
        #     print(key, value)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "graph_learner" not in name:
                param.requires_grad_(False)

        for param in self.model.graph_learner.parameters():
            param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.graph_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.float()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(model=self.model.graph_learner, optim_cfg=cfg.OPTIM)
    #    , optim_cfg, param_groups=None
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("graph_learner", self.model.graph_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # device_ids = [7]
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        # print("========prec", prec)
    
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

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
        model_file = "model.pth.tar-15"
        
        # print(epoch)

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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            # Ignore base features
            if "base_text_features" in state_dict:
                del state_dict["base_text_features"]

            if "base_img_features" in state_dict:
                del state_dict["base_img_features"]

            # Ignore base features
            if "GCN_tt.gcn_bias" in state_dict:
                del state_dict["GCN_tt.gcn_bias"]
            
            # if "graph_learner.GCN_tt.gcn_weights" in state_dict:
            #     del state_dict["graph_learner.GCN_tt.gcn_weights"]

            if "GCN_it.gcn_bias" in state_dict:
                del state_dict["GCN_it.gcn_bias"]

            # if "graph_learner.GCN_it.gcn_weights" in state_dict:
            #     del state_dict["graph_learner.GCN_it.gcn_weights"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
