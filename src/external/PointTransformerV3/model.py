"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath
from collections import OrderedDict

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .serialization import encode


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class _SharedSerializedAttention(SerializedAttention):
    """Serialized self-attention with shared QKV weights across all classes.

    Input ``point.feat`` is ``[N*C, d_head]`` — C classes stacked as a virtual
    batch.  Virtual-batch metadata must be pre-computed by
    ``FineRefinementTransformer.forward`` and stored in:

    * ``point["ccat_offset"]``  — replicated offset  [C*B]
    * ``point["ccat_order"]``   — replicated order   [C*N]
    * ``point["ccat_inverse"]`` — replicated inverse [C*N]

    Padding cache uses "ccat_*" keys to avoid collision with the decoder's
    already-cached "pad"/"unpad"/"cu_seqlens_key" entries.
    """

    def __init__(
        self,
        d_head: int,
        num_heads: int,
        patch_size: int,
        order_index: int = 0,
        enable_flash: bool = True,
    ):
        super().__init__(
            channels=d_head,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=False,
            enable_rpe=False,
            enable_flash=enable_flash,
            upcast_attention=False,
            upcast_softmax=False,
            order_index=order_index,
        )
        # Zero-init output projection → module is identity at initialisation
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        """Same logic as parent but uses point["ccat_offset"] and "ccat_*" cache keys."""
        pad_key = "ccat_pad"
        unpad_key = "ccat_unpad"
        cu_seqlens_key = "ccat_cu_seqlens"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point["ccat_offset"]   # virtual batch offset [C*B]
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point["ccat_offset"]).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels   # = d_head

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # Use pre-computed virtual batch order/inverse
        order = point["ccat_order"][pad]
        inverse = unpad[point["ccat_inverse"]]

        # Shared QKV — same weights applied to every class's d_head-dim vector
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)

        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class FineRefinementTransformer(PointModule):
    """Post-decoder refinement module combining optional spatial and class aggregation.

    Inspired by CAT-Seg's two-stage cost aggregation. Applied after the PTv3 decoder,
    where ``point.feat`` already holds per-class predictions of shape ``[N, C*E]``.

    **Stage 1 — Spatial aggregation** (``spatial_layers > 0``):
        Each class independently aggregates information from spatially nearby points,
        using PTv3's serialization-based windowed attention (patch_size tokens per window).

        Virtual-batch trick: C classes are stacked as C independent "virtual scenes",
        giving a flat ``[C*N, D]`` token sequence. Class c occupies rows ``[c*N, (c+1)*N)``.
        All C copies share the same QKV weights (``_SharedSerializedAttention``), so the
        spatial reasoning kernel is learned once and applied identically to every class.

        ``random_order=True`` samples a different serialization curve (Z / Z-trans /
        Hilbert / Hilbert-trans) each training step, preventing memorization of fixed
        neighbor windows. Inference always uses curve 0.

        ``spatial_class_chunk=K`` processes K classes per attention call instead of all C
        at once, reducing peak virtual-batch memory from C*N to K*N tokens at the cost of
        ceil(C/K) sequential attention passes. Set K=0 (default) to disable chunking.

    **Stage 2 — Class aggregation** (``class_layers > 0``):
        At each point independently, the C class tokens attend to each other via standard
        MHA (N as batch dimension, C as sequence length). This captures inter-class
        co-occurrence and mutual-exclusion patterns (e.g. "wall implies floor nearby")
        that generalize across scenes, unlike spatial patterns which are scene-specific.

    **Embedding**:
        Input ``[N, C*E]`` is first projected to ``[N, C, D]`` via a shared
        ``Linear(E, D)`` applied to each class slot independently.
        - ``E=1`` (scalar logits): each class logit is lifted to a D-dim vector.
        - ``E>1`` (rich decoder embeddings): each E-dim class embedding is upscaled to D.

    **Output**:
        ``Linear(D, 1)`` collapses the D-dim representation back to a scalar delta logit
        per class. Zero-initialized → output is 0 at init, so ``point.feat`` is unchanged
        at the start of training (identity behaviour for ``E=1`` with the residual add).

    All output projections (``spatial_fc2s``, ``class_fc2s``, ``out_proj``,
    ``class_attns[*].out_proj``) are zero-initialized → the full module is an identity
    transformation at init, making it safe to insert into a pre-trained PTv3 checkpoint.
    """

    def __init__(
        self,
        num_classes: int,
        # Latent dimension per class token. Each class is represented as a D-dim vector.
        d_head: int = 16,
        # Input channels per class from the decoder. E=1 means scalar logits;
        # E>1 means the decoder was configured with dec_channels[0] = C * E.
        embedding_dim: int = 1,
        # Number of spatial transformer layers (0 = disabled).
        spatial_layers: int = 1,
        # Number of class transformer layers (0 = disabled).
        class_layers: int = 0,
        # Number of attention heads for both spatial and class MHA.
        num_heads: int = 1,
        # FFN hidden-dim multiplier for spatial layers: hidden = D * spatial_mlp_ratio.
        spatial_mlp_ratio: int = 2,
        # FFN hidden-dim multiplier for class layers: hidden = D * class_mlp_ratio.
        class_mlp_ratio: int = 4,
        # Serialized attention window size for the spatial stage (tokens per patch).
        patch_size: int = 1024,
        enable_flash: bool = True,
        # If True, sample a random serialization curve each training step.
        random_order: bool = False,
        # Process this many classes per spatial attention call (0 = all C at once).
        # Reduces peak memory from C*N to K*N virtual tokens at the cost of ceil(C/K) passes.
        spatial_class_chunk: int = 0,
    ):
        super().__init__()
        assert spatial_layers > 0 or class_layers > 0, (
            "FineRefinementTransformer: at least one of spatial_layers or class_layers must be > 0"
        )
        C, D = num_classes, d_head
        self.num_classes = C        # C: number of classes
        self.d_head = D             # D: latent dimension per class token
        self.embedding_dim = embedding_dim  # E: decoder channels per class
        self.random_order = random_order
        self.spatial_class_chunk = spatial_class_chunk

        # Embed: Linear(E, D) applied independently to each class slot.
        # Maps [N, C, E] → [N, C, D].  Shared weights across all C classes.
        self.embed = nn.Linear(embedding_dim, D)
        self.act = nn.GELU()

        # --- Spatial aggregation blocks ---
        # Pre-norm transformer layers operating on the [C*N, D] virtual batch.
        # Each layer: LayerNorm → SerializedAttention → residual → LayerNorm → FFN → residual.
        # spatial_fc2 is zero-initialized so each layer is identity at init.
        self.spatial_norms1 = nn.ModuleList([nn.LayerNorm(D) for _ in range(spatial_layers)])
        self.spatial_attns  = nn.ModuleList([
            # _SharedSerializedAttention: PTv3 windowed attention reading from ccat_* metadata.
            # QKV weights are shared; all C class copies use the same spatial kernel.
            _SharedSerializedAttention(D, num_heads, patch_size, order_index=0, enable_flash=enable_flash)
            for _ in range(spatial_layers)
        ])
        self.spatial_norms2 = nn.ModuleList([nn.LayerNorm(D) for _ in range(spatial_layers)])
        sp_hidden = int(D * spatial_mlp_ratio)  # FFN hidden dim for spatial stage
        self.spatial_fc1s = nn.ModuleList([nn.Linear(D, sp_hidden) for _ in range(spatial_layers)])
        self.spatial_fc2s = nn.ModuleList([nn.Linear(sp_hidden, D) for _ in range(spatial_layers)])
        for fc2 in self.spatial_fc2s:
            nn.init.zeros_(fc2.weight)
            nn.init.zeros_(fc2.bias)

        # --- Class aggregation blocks ---
        # Pre-norm transformer layers where N is the batch dimension and C is the sequence.
        # At each point, all C class tokens attend to each other (full C×C attention).
        # class_attns[*].out_proj and class_fc2s are zero-initialized → identity at init.
        self.class_norms1 = nn.ModuleList([nn.LayerNorm(D) for _ in range(class_layers)])
        self.class_attns  = nn.ModuleList([
            # Standard MHA: input [N, C, D], batch_first=True → output [N, C, D].
            nn.MultiheadAttention(D, num_heads, bias=True, batch_first=True)
            for _ in range(class_layers)
        ])
        for attn in self.class_attns:
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)
        self.class_norms2 = nn.ModuleList([nn.LayerNorm(D) for _ in range(class_layers)])
        cl_hidden = int(D * class_mlp_ratio)  # FFN hidden dim for class stage
        self.class_fc1s = nn.ModuleList([nn.Linear(D, cl_hidden) for _ in range(class_layers)])
        self.class_fc2s = nn.ModuleList([nn.Linear(cl_hidden, D) for _ in range(class_layers)])
        for fc2 in self.class_fc2s:
            nn.init.zeros_(fc2.weight)
            nn.init.zeros_(fc2.bias)

        # Output projection: [N, C, D] → [N, C, 1] → squeeze → [N, C] delta logits.
        # Zero-initialized: module outputs 0 at init → point.feat unchanged (identity).
        self.out_proj = nn.Linear(D, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, point):
        # point.feat: [N, C*E]
        #   N = total voxelized points across all scenes in the batch (PTv3 flat layout)
        #   C = num_classes, E = embedding_dim
        N, feat_C = point.feat.shape
        C, D, E = self.num_classes, self.d_head, self.embedding_dim
        assert feat_C == C * E, f"FineRefinementTransformer: expected [N, {C*E}], got [N, {feat_C}]"

        # --- Embed input to latent space ---
        if E == 1:
            orig_feat = point.feat                        # [N, C]   saved for output residual
            x = self.embed(orig_feat.unsqueeze(-1))       # [N, C, 1] → Linear(1, D) → [N, C, D]
        else:
            # Rich decoder embeddings: reshape to expose the E-dim per class, then project.
            x = self.embed(point.feat.reshape(N, C, E))  # [N, C, E] → Linear(E, D) → [N, C, D]

        # --- Spatial aggregation ---
        # Virtual-batch trick: treat each class as an independent copy of the scene.
        # Class c occupies rows [c*N, (c+1)*N) in the flat virtual sequence.
        # Shared QKV weights mean all classes use the same spatial attention kernel.
        if self.spatial_norms1:
            # Choose which serialization curve to use.
            # point.serialized_order is a list of len=num_orders permutation tensors, each [N].
            if self.random_order and self.training:
                # Sample uniformly from available curves to prevent memorizing fixed windows.
                order_idx = torch.randint(len(point.serialized_order), (1,)).item()
            else:
                order_idx = 0  # Always use curve 0 at inference for determinism.

            orig_offset  = point.offset                         # [B] cumulative point counts
            bincount     = offset2bincount(orig_offset)         # [B] points per scene
            orig_order   = point.serialized_order[order_idx]    # [N] permutation: sorted→orig
            orig_inverse = point.serialized_inverse[order_idx]  # [N] permutation: orig→sorted

            chunk = self.spatial_class_chunk  # K classes per attention call; 0 = all C
            use_chunks = 0 < chunk < C

            if not use_chunks:
                # Build the full C*N virtual batch metadata once.
                # virt_offset: [B*C] — cumulative point counts for the virtual batch.
                #   Each actual scene is replicated C times: [scene0-c0, scene1-c0, scene0-c1, ...]
                virt_offset  = torch.cumsum(bincount.repeat(C), dim=0)          # [B*C]
                # virt_order: [C*N] — serialization order for the virtual batch.
                #   Class c's points sit at virtual indices [c*N, (c+1)*N), so shift orig_order
                #   by c*N to keep them within their class's address range.
                virt_order   = torch.cat([orig_order   + c * N for c in range(C)])  # [C*N]
                # virt_inverse: [C*N] — maps each virtual point back to its sorted position.
                virt_inverse = torch.cat([orig_inverse + c * N for c in range(C)])  # [C*N]
                # Write metadata; _SharedSerializedAttention reads from ccat_* keys.
                # Cache is valid for all spatial layers (same N and C throughout).
                point["ccat_offset"]  = virt_offset
                point["ccat_order"]   = virt_order
                point["ccat_inverse"] = virt_inverse

            for norm1, attn, norm2, fc1, fc2 in zip(
                self.spatial_norms1, self.spatial_attns,
                self.spatial_norms2, self.spatial_fc1s, self.spatial_fc2s,
            ):
                # Pre-norm attention sublayer
                shortcut = x
                x_normed = norm1(x)   # [N, C, D]

                if not use_chunks:
                    # Flatten to [C*N, D]: class c occupies rows [c*N, (c+1)*N).
                    # permute(1,0,2): [C, N, D]; reshape: [C*N, D].
                    point.feat = x_normed.permute(1, 0, 2).reshape(C * N, D)  # [C*N, D]
                    point = attn(point)   # serialized attention → point.feat: [C*N, D]
                    # Unflatten: reshape [C*N, D] → [C, N, D]; permute → [N, C, D].
                    x = shortcut + point.feat.reshape(C, N, D).permute(1, 0, 2)  # [N, C, D]
                else:
                    # Chunked path: process K classes at a time to cap peak memory at K*N tokens.
                    attn_out = torch.empty_like(x_normed)  # [N, C, D] output buffer
                    for c_start in range(0, C, chunk):
                        c_end = min(c_start + chunk, C)
                        K = c_end - c_start  # actual chunk size (last chunk may be smaller)
                        x_chunk = x_normed[:, c_start:c_end, :]           # [N, K, D]
                        # Build K-class virtual batch metadata (same logic as full path).
                        virt_offset_k  = torch.cumsum(bincount.repeat(K), dim=0)           # [B*K]
                        virt_order_k   = torch.cat([orig_order   + k * N for k in range(K)])  # [K*N]
                        virt_inverse_k = torch.cat([orig_inverse + k * N for k in range(K)])  # [K*N]
                        # Clear cached pad/unpad/cu_seqlens from the previous chunk;
                        # _SharedSerializedAttention caches these based on ccat_offset,
                        # so they must be invalidated whenever the virtual batch changes.
                        for key in ("ccat_pad", "ccat_unpad", "ccat_cu_seqlens"):
                            if key in point.keys():
                                del point[key]
                        point["ccat_offset"]  = virt_offset_k
                        point["ccat_order"]   = virt_order_k
                        point["ccat_inverse"] = virt_inverse_k
                        point.feat = x_chunk.permute(1, 0, 2).reshape(K * N, D)  # [K*N, D]
                        point = attn(point)   # → point.feat: [K*N, D]
                        attn_out[:, c_start:c_end, :] = point.feat.reshape(K, N, D).permute(1, 0, 2)
                    x = shortcut + attn_out  # [N, C, D]

                # Pre-norm FFN sublayer (shared weights, applied point-wise across N*C slots)
                shortcut = x
                x_normed = norm2(x)                                              # [N, C, D]
                x_flat = x_normed.reshape(N * C, D)                             # [N*C, D]
                x = shortcut + fc2(self.act(fc1(x_flat))).reshape(N, C, D)      # [N, C, D]

        # --- Class aggregation ---
        # Standard MHA with N as the batch dimension and C as the sequence dimension.
        # Each point independently mixes its C class tokens → captures which classes
        # co-occur or exclude each other at any given spatial location.
        for norm1, attn, norm2, fc1, fc2 in zip(
            self.class_norms1, self.class_attns,
            self.class_norms2, self.class_fc1s, self.class_fc2s,
        ):
            # Pre-norm attention sublayer: full C×C self-attention per point
            shortcut = x
            x_normed = norm1(x)                                    # [N, C, D]
            attn_out, _ = attn(x_normed, x_normed, x_normed)      # [N, C, D]
            x = shortcut + attn_out                                # [N, C, D]

            # Pre-norm FFN sublayer (weights shared across N and C)
            shortcut = x
            x_normed = norm2(x)                                    # [N, C, D]
            x_flat = x_normed.reshape(N * C, D)                   # [N*C, D]
            x = shortcut + fc2(self.act(fc1(x_flat))).reshape(N, C, D)  # [N, C, D]

        # --- Output ---
        # Project each D-dim class token to a scalar delta logit: [N, C, D] → [N, C, 1] → [N, C].
        logits = self.out_proj(x).squeeze(-1)   # [N, C]
        # E=1: add delta logits to the original input logits (residual connection).
        # E>1: output the full [N, C] predictions directly (no residual; decoder handles skip).
        point.feat = orig_feat + logits if E == 1 else logits
        return point


class ClassAggregationTransformer(PointModule):
    """Inter-class attention inserted between the PTv3 encoder and decoder (bottleneck).

    **Motivation**: The encoder bottleneck is the most spatially compressed stage
    (fewest points, largest receptive field). Mixing class information here propagates
    class co-occurrence context through all subsequent decoder skip connections,
    making it available at every resolution during upsampling.

    **Shape constraint**: Requires ``enc_channels[-1] = num_classes * d_head`` so the
    bottleneck features ``[N_coarse, C*D]`` can be freely reshaped to ``[N_coarse, C, D]``
    without a learned projection. ``d_head`` is inferred at build time as
    ``enc_channels[-1] // num_classes``.

    **Attention layout**: ``N_coarse`` is treated as the batch dimension and ``C`` as the
    sequence dimension. Each coarse point independently mixes its C class-slot features
    via full C×C self-attention. No positional encoding is added — the module is
    permutation-invariant over class ordering, which is consistent with the class ordering
    varying per scene (PTv3 sorts classes by spatial frequency).

    **Data flow**::

        [N_coarse, C*D]  →  reshape  →  [N_coarse, C, D]
                         →  L × (MHA + FFN)
                         →  reshape  →  [N_coarse, C*D]

    The modified ``[N_coarse, C*D]`` tensor is then passed to the PTv3 decoder, which
    upsamples it via skip connections at each decoder stage.

    **Identity initialisation**: ``attn.out_proj`` and ``fc2`` weights/biases are all
    zero-initialized → module is an exact identity at init, so inserting it into a
    pretrained PTv3 checkpoint causes no immediate degradation.
    """

    def __init__(
        self,
        num_classes: int,
        # Latent dimension per class token = enc_channels[-1] // num_classes.
        # Inferred at build time in build_model_from_cfg; not a free hyperparameter.
        d_head: int,
        # Number of attention heads for MHA. Must divide d_head evenly.
        num_heads: int = 1,
        # Number of stacked MHA + FFN layers.
        num_layers: int = 1,
        # FFN hidden-dim multiplier: hidden = d_head * mlp_ratio.
        mlp_ratio: int = 4,
    ):
        super().__init__()
        C, D = num_classes, d_head
        self.num_classes = C  # C: number of classes (sequence length for MHA)
        self.d_head = D       # D: feature dimension per class token

        # Pre-norm MHA sublayer: LayerNorm(D) → MHA(D, num_heads) → residual.
        self.norms1 = nn.ModuleList([nn.LayerNorm(D) for _ in range(num_layers)])
        self.attns = nn.ModuleList([
            # batch_first=True: input/output shape [N_coarse, C, D].
            nn.MultiheadAttention(D, num_heads, bias=True, batch_first=True)
            for _ in range(num_layers)
        ])
        # Zero-init out_proj → attention sublayer is identity at init.
        for attn in self.attns:
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)

        # Pre-norm FFN sublayer: LayerNorm(D) → Linear(D, hidden) → GELU → Linear(hidden, D) → residual.
        self.norms2 = nn.ModuleList([nn.LayerNorm(D) for _ in range(num_layers)])
        hidden = int(D * mlp_ratio)  # FFN hidden dimension
        self.fc1s = nn.ModuleList([nn.Linear(D, hidden) for _ in range(num_layers)])
        self.fc2s = nn.ModuleList([nn.Linear(hidden, D) for _ in range(num_layers)])
        # Zero-init fc2 → FFN sublayer is identity at init.
        for fc2 in self.fc2s:
            nn.init.zeros_(fc2.weight)
            nn.init.zeros_(fc2.bias)
        self.act = nn.GELU()

    def forward(self, point):
        # point.feat: [N_coarse, C*D]
        #   N_coarse = voxelized points at the encoder bottleneck resolution
        #              (much smaller than the fine-resolution N after decoding)
        #   C = num_classes, D = d_head
        N, feat_C = point.feat.shape
        C, D = self.num_classes, self.d_head
        assert feat_C == C * D, (
            f"ClassAggregationTransformer: expected point.feat [N, {C * D}], got [N, {feat_C}]. "
            f"Set enc_channels[-1] = class_subset * D ({C} * {D} = {C * D})."
        )

        # Reshape to expose C class tokens per point: [N_coarse, C*D] → [N_coarse, C, D].
        # N_coarse acts as the MHA batch dimension; C is the sequence length.
        x = point.feat.reshape(N, C, D)   # [N_coarse, C, D]

        for norm1, attn, norm2, fc1, fc2 in zip(
            self.norms1, self.attns, self.norms2, self.fc1s, self.fc2s
        ):
            # Pre-norm self-attention: each coarse point mixes all C class tokens.
            shortcut = x
            x_normed = norm1(x)                                    # [N_coarse, C, D]
            attn_out, _ = attn(x_normed, x_normed, x_normed)      # [N_coarse, C, D]
            x = shortcut + attn_out                                # [N_coarse, C, D]

            # Pre-norm FFN: Linear(D→hidden)→GELU→Linear(hidden→D), shared across N_coarse and C.
            shortcut = x
            x_normed = norm2(x)                                    # [N_coarse, C, D]
            x_flat = x_normed.reshape(N * C, D)                   # [N_coarse*C, D]
            x = shortcut + fc2(self.act(fc1(x_flat))).reshape(N, C, D)  # [N_coarse, C, D]

        # Flatten back: [N_coarse, C, D] → [N_coarse, C*D] for the PTv3 decoder.
        point.feat = x.reshape(N, C * D)
        return point


class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        enable_skip=False,
        skip_to_output=False,
        num_classes=100,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        fine_refinement=None,
        class_aggregator=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.enable_skip = enable_skip
        self.skip_to_output = skip_to_output
        self.num_classes = num_classes
        self.feat_channels = in_channels
        assert not (self.enable_skip and dec_channels[0] != self.num_classes), (
            "enable_skip requires dec_channels[0] == num_classes. "
            "Disable enable_skip when using embedding_dim > 1."
        )
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        # Optional fine-resolution refinement (appended after decoder)
        self.fine_refinement = None
        if fine_refinement is not None:
            fr = fine_refinement
            self.fine_refinement = FineRefinementTransformer(
                num_classes=num_classes,
                d_head=fr["d_head"],
                embedding_dim=fr["embedding_dim"],
                spatial_layers=fr["spatial_layers"],
                class_layers=fr["class_layers"],
                num_heads=fr["num_heads"],
                spatial_mlp_ratio=fr["spatial_mlp_ratio"],
                class_mlp_ratio=fr["class_mlp_ratio"],
                patch_size=fr["patch_size"],
                enable_flash=fr["enable_flash"],
                random_order=fr["random_order"],
            )

        # Optional class aggregation transformer (inserted between encoder and decoder)
        self.class_aggregator = None
        if class_aggregator is not None:
            ca = class_aggregator
            _d_head_ca = enc_channels[-1] // num_classes
            self.class_aggregator = ClassAggregationTransformer(
                num_classes=num_classes,
                d_head=_d_head_ca,
                num_heads=ca.get("num_heads", 1),
                num_layers=ca.get("num_layers", 1),
                mlp_ratio=ca.get("mlp_ratio", 4),
            )

    def forward(self, data_dict):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        data_dict["feat"] = data_dict["feat"][:,:self.feat_channels]
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        if self.enable_skip or self.skip_to_output:
            input_logits = point.feat[:, :self.num_classes].clone()

        point = self.embedding(point)
        point = self.enc(point)
        if self.class_aggregator is not None:
            point = self.class_aggregator(point)
        if not self.cls_mode:
            point = self.dec(point)
        if self.enable_skip:
            point.feat += input_logits
        if self.fine_refinement is not None:
            point = self.fine_refinement(point)
        if self.skip_to_output:
            point.feat += input_logits
        return point
