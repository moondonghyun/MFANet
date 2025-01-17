import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from kornia.color import rgb_to_raw, CFA
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import pywt
import math
import numpy as np
import torch_dct as dct
import cv2
def depth_wise_conv(in_dim,out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=False, groups=in_dim),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
    )

def upsampling(scale_factor, in_dim,out_dim):
    return nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=scale_factor),
        nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=False, groups=in_dim),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
    )
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
def one_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Squeeze_And_Excitation(nn.Module):
    def __init__(self,in_channels,r=16):
        super(Squeeze_And_Excitation,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels,in_channels//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//r,in_channels),
            #nn.Sigmoid()
        )
    def forward(self,x):
        x = self.squeeze(x)
        x = x.view(x.size(0),-1)
        x = self.excitation(x)
        x = x.view(x.size(0),x.size(1),1,1)
        return x

class PositionAttentionModule(nn.Module):
    ''' self-attention '''

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """

        N, C, H, W = x.shape
        query = self.query_conv(x).view(
            N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
        key = self.key_conv(x).view(N, -1, H*W)  # (N, C', H*W)

        # caluculate correlation
        energy = torch.bmm(query, key)    # (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H*W)    # (N, C, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out

class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out

class Res_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class AttentionMap(nn.Module):
    def __init__(self, in_channels):
        super(AttentionMap, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        attention_map = self.conv1x1(x) 
        
        return attention_map
class Squeeze_And_Excitation(nn.Module):
    def __init__(self,in_channels,r=8):
        super(Squeeze_And_Excitation,self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels,in_channels//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//r,in_channels),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.squeeze(x)
        x = x.view(x.size(0),-1)
        x = self.excitation(x)
        x = x.view(x.size(0),x.size(1),1,1)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i == 0:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                self.skip_block1 = x
            if i == 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                self.skip_block2 = x
            if i == 2:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                self.skip_block3 = x
            if i == self.num_stages - 1:
                t = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                self.skip_block4 = t

        return x.mean(dim=1),self.skip_block1, self.skip_block2, self.skip_block3, self.skip_block4

    def forward(self, x):
        x,skip1,skip2,skip3,skip4 = self.forward_features(x)
        x = self.head(x)

        return skip1,skip2,skip3,skip4

def setup_srm_weights(input_channels: int = 3, output_channel=1) -> torch.Tensor:
    srm_kernel = torch.from_numpy(
        np.array([
            [  # srm 1/2 horiz
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 1., -2., 1., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
            ],
            [  # srm 1/4
                [0., 0., 0., 0., 0.],
                [0., -1., 2., -1., 0.],
                [0., 2., -4., 2., 0.],
                [0., -1., 2., -1., 0.],
                [0., 0., 0., 0., 0.],
            ],
            [  # srm 1/12
                [-1., 2., -2., 2., -1.],
                [2., -6., 8., -6., 2.],
                [-2., 8., -12., 8., -2.],
                [2., -6., 8., -6., 2.],
                [-1., 2., -2., 2., -1.],
            ]
        ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(output_channel, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3, output_channel=None) -> torch.nn.Module:
    if output_channel == None:
        weights = setup_srm_weights(input_channels)
        conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    else:
        weights = setup_srm_weights(input_channels, output_channel)
        conv = torch.nn.Conv2d(input_channels,
                               out_channels=output_channel,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights, requires_grad=False)
    return conv

# class TextEmbedding(nn.Module):
#     def __init__(self, embedding_dim=64):
#         super(TextEmbedding, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.linear1 = None
#         self.linear2 = None
#         self.relu = nn.ReLU(inplace = True)

#     def forward(self, embedding_vector, batch_size):
#         if self.linear1 is None:  # 최초 forward 통과 시에만 Linear 레이어를 생성
#             self.linear1 = nn.Linear(self.embedding_dim, batch_size*self.embedding_dim).to('cuda')
#             self.linear2 = nn.Linear(batch_size*self.embedding_dim, batch_size*self.embedding_dim).to('cuda')
        
#         if batch_size != 16:
#             self.linear1 = nn.Linear(self.embedding_dim, batch_size*self.embedding_dim).to('cuda')
#             self.linear2 = nn.Linear(batch_size*self.embedding_dim, batch_size*self.embedding_dim).to('cuda')
#         print(batch_size)
#         x = self.linear1(embedding_vector)  # [batch_size, embedding_dim]
#         x = self.relu(x)
#         x = self.linear2(x)
        
#         # 변환된 결과를 [batch_size, embedding_dim, 1, 1]로 변환
#         x = x.view(batch_size, self.embedding_dim, 1, 1)
#         return x

        
class Decoder_Module(nn.Module):
    def __init__(self,embedding_dim=768,threshold=7.3):
        super(Decoder_Module,self).__init__()

        self.pvtv2_b2_RGB = pvt_v2_RGB()

        self.rgb_skip_layer1 = nn.Conv2d(576,64,1)
        self.rgb_skip_layer2 = nn.Conv2d(384,64,1)
        self.rgb_skip_layer3 = nn.Conv2d(192,64,1)
        self.rgb_skip_layer4 = nn.Conv2d(64,64,1)

        self.PA1_LL = PositionAttentionModule(64)
        self.PA1_LL_conv = one_conv(64,64)

        self.PA1_LH = PositionAttentionModule(64)
        self.PA1_LH_conv = one_conv(64,64)

        self.PA1_HL = PositionAttentionModule(64)
        self.PA1_HL_conv = one_conv(64,64)

        self.PA1_HH = PositionAttentionModule(64)
        self.PA1_HH_conv = one_conv(64,64)

        self.PA2_LL = PositionAttentionModule(64)
        self.PA2_LL_conv = one_conv(64,64)

        self.PA2_LH = PositionAttentionModule(64)
        self.PA2_LH_conv = one_conv(64,64)

        self.PA2_HL = PositionAttentionModule(64)
        self.PA2_HL_conv = one_conv(64,64)

        self.PA2_HH = PositionAttentionModule(64)
        self.PA2_HH_conv = one_conv(64,64)

        self.PA3_LL = PositionAttentionModule(64)
        self.PA3_LL_conv = one_conv(64,64)

        self.PA3_LH = PositionAttentionModule(64)
        self.PA3_LH_conv = one_conv(64,64)

        self.PA3_HL = PositionAttentionModule(64)
        self.PA3_HL_conv = one_conv(64,64)

        self.PA3_HH = PositionAttentionModule(64)
        self.PA3_HH_conv = one_conv(64,64)

        self.PA4_LL = PositionAttentionModule(64)
        self.PA4_LL_conv = one_conv(64,64)

        self.PA4_LH = PositionAttentionModule(64)
        self.PA4_LH_conv = one_conv(64,64)

        self.PA4_HL = PositionAttentionModule(64)
        self.PA4_HL_conv = one_conv(64,64)

        self.PA4_HH = PositionAttentionModule(64)
        self.PA4_HH_conv = one_conv(64,64)

        self.upsampling1 = upsampling(2,64,64)
        self.upsampling2 = upsampling(2,128,64)
        self.upsampling3 = upsampling(2,128,64)
        self.upsampling4 = upsampling(2,128,64)
        self.upsampling5 = upsampling(2,64,1)

        # self.out_upsampling2 = upsampling(16,128,32)
        # self.out_upsampling3 = upsampling(8,128,32)

        self.att1_LH = AttentionMap(128)
        self.att1_HL = AttentionMap(128)
        self.att1_HH = AttentionMap(128)

        self.att2_LH = AttentionMap(128)
        self.att2_HL = AttentionMap(128)
        self.att2_HH = AttentionMap(128)

        self.att3_LH = AttentionMap(128)
        self.att3_HL = AttentionMap(128)
        self.att3_HH = AttentionMap(128)

        self.att4_LH = AttentionMap(128)
        self.att4_HL = AttentionMap(128)
        self.att4_HH = AttentionMap(128)

        self.dconv1_1 = Res_block(64,32)
        self.dconv1_2 = Res_block(64,32)

        self.dconv2_1 = Res_block(128,64)
        self.dconv2_2 = Res_block(128,64)

        self.dconv3_1 = Res_block(128,64)
        self.dconv3_2 = Res_block(128,64)

        self.CA1 = Squeeze_And_Excitation(64)
        self.CA2 = Squeeze_And_Excitation(64)
        self.CA3 = Squeeze_And_Excitation(64)
        self.CA4 = Squeeze_And_Excitation(64)

        self.dconv4_1 = Res_block(128,64)
        self.dconv4_2 = Res_block(128,64)

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.out_conv = nn.Conv2d(128,1,1)

        self.vocab = ["easy", "hard"]
        self.threshold = threshold
        self.embedding = nn.Embedding(len(self.vocab), embedding_dim)
        self.word_to_index = {word : idx for idx,word in enumerate(self.vocab)}

        self.attention1 = nn.Sequential(
            nn.Linear(embedding_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Linear(embedding_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.attention3 = nn.Sequential(
            nn.Linear(embedding_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.attention4 = nn.Sequential(
            nn.Linear(embedding_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def calculate_curvature(self, label_mask):
        """
        곡률 계산 함수. Sobel 필터를 사용하여 경계 추출 후 곡률 계산.
        """
        label_mask_np = label_mask.squeeze().detach().cpu().numpy()
        edges = cv2.Canny((label_mask_np * 255).astype(np.uint8), 100, 200)
        points = np.column_stack(np.where(edges > 0))

        if len(points) < 3:
            return 0

        curvatures = []

        for i in range(1, len(points) - 1):
            prev_point = points[i - 1]
            curr_point = points[i]
            next_point = points[i + 1]

            v1 = curr_point - prev_point
            v2 = next_point - curr_point
            curvature = np.linalg.norm(v2 - v1)
            curvatures.append(curvature)

        return np.mean(curvatures)

    def classify_curvature(self, curvature):
        return "hard" if curvature > self.threshold else "easy"

    def forward(self,x):
        #([1, 64,64,64]), ([1, 128, 32, 32]), ([1, 320, 16, 16]), ([1, 512, 8, 8])
        RGB_block4,RGB_block3,RGB_block2,RGB_block1 = self.pvtv2_b2_RGB(x)

        rgb_block4 = self.rgb_skip_layer4(RGB_block4) # 64

        LL_comb_block4,(LH_comb_block4, HL_comb_block4,HH_comb_block4) = pywt.dwt2(rgb_block4.detach().cpu(),wavelet='haar') # 64 32 32

        LL_comb_block4 = torch.tensor(LL_comb_block4, device='cuda', dtype=torch.float32)
        LH_comb_block4 = torch.tensor(LH_comb_block4, device='cuda', dtype=torch.float32)
        HL_comb_block4 = torch.tensor(HL_comb_block4, device='cuda', dtype=torch.float32)
        HH_comb_block4 = torch.tensor(HH_comb_block4, device='cuda', dtype=torch.float32)
        
        rgb_block3 = torch.cat([LL_comb_block4,RGB_block3],dim=1) # 128 + 64
        rgb_block3 = self.rgb_skip_layer3(rgb_block3)

        LL_comb_block3,(LH_comb_block3, HL_comb_block3,HH_comb_block3) = pywt.dwt2(rgb_block3.detach().cpu(),wavelet='haar') # 64 16 16

        LL_comb_block3 = torch.tensor(LH_comb_block3, device='cuda', dtype=torch.float32)
        LH_comb_block3 = torch.tensor(LH_comb_block3, device='cuda', dtype=torch.float32)
        HL_comb_block3 = torch.tensor(HL_comb_block3, device='cuda', dtype=torch.float32)
        HH_comb_block3 = torch.tensor(HH_comb_block3, device='cuda', dtype=torch.float32)

        rgb_block2 = torch.cat([LL_comb_block3,RGB_block2],dim=1) # 320 + 64
        rgb_block2 = self.rgb_skip_layer2(rgb_block2)

        LL_comb_block2,(LH_comb_block2, HL_comb_block2,HH_comb_block2) = pywt.dwt2(rgb_block2.detach().cpu(),wavelet='haar') # 64 8 8 

        LL_comb_block2 = torch.tensor(LL_comb_block2, device='cuda', dtype=torch.float32)
        LH_comb_block2 = torch.tensor(LH_comb_block2, device='cuda', dtype=torch.float32)
        HL_comb_block2 = torch.tensor(HL_comb_block2, device='cuda', dtype=torch.float32)
        HH_comb_block2 = torch.tensor(HH_comb_block2, device='cuda', dtype=torch.float32)


        rgb_block1 = torch.cat([LL_comb_block2,RGB_block1],dim=1) # 512 + 64
        rgb_block1 = self.rgb_skip_layer1(rgb_block1)

        LL_comb_block1,(LH_comb_block1, HL_comb_block1,HH_comb_block1) = pywt.dwt2(rgb_block1.detach().cpu(),wavelet='haar')

        LH_comb_block1 = torch.tensor(LH_comb_block1, device='cuda', dtype=torch.float32)
        HL_comb_block1 = torch.tensor(HL_comb_block1, device='cuda', dtype=torch.float32)
        HH_comb_block1 = torch.tensor(HH_comb_block1, device='cuda', dtype=torch.float32)

        #label detection
        PA1_LH = self.PA1_LH(LH_comb_block1)
        PA1_LH = self.PA1_LH_conv(PA1_LH)

        PA1_HL = self.PA1_HL(HL_comb_block1)
        PA1_HL = self.PA1_HL_conv(PA1_HL)

        PA1_HH = self.PA1_HH(HH_comb_block1)
        PA1_HH = self.PA1_HH_conv(PA1_HH)
        
        LH1_ = self.upsampling(PA1_LH)
        HL1_ = self.upsampling(PA1_HL)
        HH1_ = self.upsampling(PA1_HH)

        LH1 = torch.cat([rgb_block1,LH1_],dim=1)
        HL1 = torch.cat([rgb_block1,HL1_],dim=1)
        HH1 = torch.cat([rgb_block1,HH1_],dim=1)

        att1_LH = self.att1_LH(LH1)
        att1_HL = self.att1_HL(HL1)
        att1_HH = self.att1_HH(HH1)
        
        rgb_block1 = rgb_block1 * self.sigmoid(self.CA1(rgb_block1) * (att1_LH + att1_HL + att1_HH))
        
        x = self.dconv1_1(rgb_block1)
        x = self.dconv1_2(rgb_block1)
        x = self.upsampling1(x)

        PA2_LH = self.PA2_LH(LH_comb_block2)
        PA2_LH = self.PA2_LH_conv(PA2_LH)

        PA2_HL = self.PA2_HL(HL_comb_block2)
        PA2_HL = self.PA2_HL_conv(PA2_HL)

        PA2_HH = self.PA2_HH(HH_comb_block2)
        PA2_HH = self.PA2_HH_conv(PA2_HH)

        LH2_ = self.upsampling(PA2_LH)
        HL2_ = self.upsampling(PA2_HL)
        HH2_ = self.upsampling(PA2_HH)

        LH2 = torch.cat([rgb_block2,LH2_],dim=1)
        HL2 = torch.cat([rgb_block2,HL2_],dim=1)
        HH2 = torch.cat([rgb_block2,HH2_],dim=1)

        att2_LH = self.att2_LH(LH2)
        att2_HL = self.att2_HL(HL2)
        att2_HH = self.att2_HH(HH2)

        rgb_block2 = rgb_block2 * self.sigmoid(self.CA2(rgb_block2) * (att2_LH + att2_HL + att2_HH))

        x = torch.cat([x,rgb_block2],dim=1)
        x = self.dconv2_1(x)
        x = self.dconv2_2(x)

        # out_16x = self.out_upsampling2(x)
        x = self.upsampling2(x)

        PA3_LH = self.PA3_LH(LH_comb_block3)
        PA3_LH = self.PA3_LH_conv(PA3_LH)

        PA3_HL = self.PA3_HL(HL_comb_block3)
        PA3_HL = self.PA3_HL_conv(PA3_HL)

        PA3_HH = self.PA3_HH(HH_comb_block3)
        PA3_HH = self.PA3_HH_conv(PA3_HH)

        LH3_ = self.upsampling(PA3_LH)
        HL3_ = self.upsampling(PA3_HL)
        HH3_ = self.upsampling(PA3_HH)

        LH3 = torch.cat([rgb_block3,LH3_],dim=1)
        HL3 = torch.cat([rgb_block3,HL3_],dim=1)
        HH3 = torch.cat([rgb_block3,HH3_],dim=1)

        att3_LH = self.att3_LH(LH3)
        att3_HL = self.att3_HL(HL3)
        att3_HH = self.att3_HH(HH3)

        rgb_block3 = rgb_block3 * self.sigmoid(self.CA3(rgb_block3) * (att3_LH + att3_HL + att3_HH))

        x = torch.cat([x,rgb_block3],dim=1)
        x = self.dconv3_1(x)
        x = self.dconv3_2(x)

        # out_8x = self.out_upsampling3(x)
        x = self.upsampling3(x)

        PA4_LH = self.PA4_LH(LH_comb_block4)
        PA4_LH = self.PA4_LH_conv(PA4_LH)

        PA4_HL = self.PA4_HL(HL_comb_block4)
        PA4_HL = self.PA4_HL_conv(PA4_HL)

        PA4_HH = self.PA4_HH(HH_comb_block4)
        PA4_HH = self.PA4_HH_conv(PA4_HH)

        LH4_ = self.upsampling(PA4_LH)
        HL4_ = self.upsampling(PA4_HL)
        HH4_ = self.upsampling(PA4_HH)

        LH4 = torch.cat([rgb_block4,LH4_],dim=1)
        HL4 = torch.cat([rgb_block4,HL4_],dim=1)
        HH4 = torch.cat([rgb_block4,HH4_],dim=1)

        att4_LH = self.att4_LH(LH4)
        att4_HL = self.att4_HL(HL4)
        att4_HH = self.att4_HH(HH4)

        rgb_block4 = rgb_block4 * self.sigmoid(self.CA4(rgb_block4) * (att4_LH + att4_HL + att4_HH))

        x = torch.cat([x,rgb_block4],dim=1)
        x = self.dconv4_1(x)
        x = self.dconv4_2(x)

        label_mask = self.upsampling4(x)
        label_mask = self.upsampling5(label_mask)

        curvature = self.calculate_curvature(label_mask)
        embedding_word= self.classify_curvature(curvature)
        print(embedding_word)
        embedding_index = torch.tensor(
        [self.word_to_index[embedding_word]],
        dtype=torch.long, device=x.device
        )
        embedding_vector = self.embedding(embedding_index)
        
        rgb_block1 = rgb_block1 * self.sigmoid((self.attention1(embedding_vector).unsqueeze(2).unsqueeze(3) +self.CA1(rgb_block1))  * (att1_LH + att1_HL + att1_HH))
        
        x = self.dconv1_1(rgb_block1)
        x = self.dconv1_2(rgb_block1)
        x = self.upsampling1(x)

        rgb_block2 = rgb_block2 * self.sigmoid((self.attention2(embedding_vector).unsqueeze(2).unsqueeze(3)+self.CA2(rgb_block2) )* (att2_LH + att2_HL + att2_HH))

        x = torch.cat([x,rgb_block2],dim=1)
        x = self.dconv2_1(x)
        x = self.dconv2_2(x)
        x = self.upsampling2(x)

        rgb_block3 = rgb_block3 * self.sigmoid((self.attention3(embedding_vector).unsqueeze(2).unsqueeze(3)+self.CA3(rgb_block3)) * (att3_LH + att3_HL + att3_HH))

        x = torch.cat([x,rgb_block3],dim=1)
        x = self.dconv3_1(x)
        x = self.dconv3_2(x)
        x = self.upsampling3(x)


        rgb_block4 = rgb_block4 * self.sigmoid((self.attention4(embedding_vector).unsqueeze(2).unsqueeze(3) +self.CA4(rgb_block4))* (att4_LH + att4_HL + att4_HH))

        x = torch.cat([x,rgb_block4],dim=1)
        x = self.dconv4_1(x)
        x = self.dconv4_2(x)

        label_mask = self.upsampling4(x)


        label_mask = self.upsampling5(label_mask)

        return label_mask

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def pvt_v2_RGB(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()
    model.load_state_dict(torch.load('../model_save/pvt_v2_b2.pth'))
    return model

def pvt_v2_FRE(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()
    model.load_state_dict(torch.load('../model_save/pvt_v2_b2.pth'))
    return model

# if __name__=='__main__':
#     model = Decoder_Module().cuda()
#     #model.load_state_dict(torch.load('../model_save/pvt_v2_b2.pth'))
#     #model = PyramidVisionTransformerV2().cuda()
#     inp = torch.randn((1, 3, 224, 224)).cuda()
#     oup= model(inp)
#     print(oup.shape)