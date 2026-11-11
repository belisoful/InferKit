"""CPU stand-in for spconv 2.x: SparseConvTensor and the submanifold SubMConv3d.

SubMConv3d keeps the active sites. For kernel offset k (flattened x-major over [kx, ky, kz]) an output
site at p reads the input site at p + (k - center) * dilation, the dense Conv3d correlation. Missing
neighbors read zero. The neighbor features are gathered into one [N, K * C_in] matrix and multiplied by
the weight [C_out, K * C_in] in a single GEMM, then the bias is added.
"""
import enum
import torch
import torch.nn as nn


class ConvAlgo(enum.Enum):
    Native = 0
    MaskImplicitGemm = 1


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None, indice_dict=None, **kwargs):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = {} if indice_dict is None else indice_dict
        self.benchmark = False
        self.benchmark_record = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = None
        self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    def replace_feature(self, features):
        new = SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size, self.grid, self.voxel_num, self.indice_dict)
        return new

    def dense(self, channels_first=True):
        feats = self._features.reshape(self._features.shape[0], -1)
        out = feats.new_zeros([self.batch_size] + self.spatial_shape + [feats.shape[1]])
        idx = self.indices.long()
        out[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = feats
        return out.permute(0, 4, 1, 2, 3).contiguous() if channels_first else out


def _neighbor_table(indices, spatial_shape, kernel_size, dilation):
    idx = indices.long()
    dims = torch.tensor([int(s) for s in spatial_shape], dtype=torch.long)
    pad = (dims + 2 * (kernel_size // 2) * dilation)
    def key(b, c):
        return ((b * pad[0] + c[:, 0]) * pad[1] + c[:, 1]) * pad[2] + c[:, 2]
    shift = (kernel_size // 2) * dilation
    keys = key(idx[:, 0], idx[:, 1:] + shift)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    r = torch.arange(kernel_size)
    offsets = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3) - kernel_size // 2
    table = torch.full((idx.shape[0], offsets.shape[0]), -1, dtype=torch.long)
    for k, off in enumerate(offsets):
        q = key(idx[:, 0], idx[:, 1:] + shift + off * dilation)
        pos = torch.searchsorted(sorted_keys, q).clamp(max=sorted_keys.numel() - 1)
        hit = sorted_keys[pos] == q
        table[:, k] = torch.where(hit, order[pos], torch.full_like(pos, -1))
    return table


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__()
        assert stride == 1
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.dilation = int(kernel_size), int(dilation)
        self.indice_key = indice_key
        k = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, k, k, k, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x):
        feats = x.features.reshape(x.features.shape[0], -1)
        cache_key = (self.indice_key, self.kernel_size, self.dilation)
        table = x.indice_dict.get(cache_key) if self.indice_key is not None else None
        if table is None:
            table = _neighbor_table(x.indices, x.spatial_shape, self.kernel_size, self.dilation)
            if self.indice_key is not None:
                x.indice_dict[cache_key] = table
        padded = torch.cat([feats, feats.new_zeros(1, feats.shape[1])])
        gathered = padded[torch.where(table >= 0, table, feats.shape[0])]
        gathered = gathered.reshape(feats.shape[0], -1)
        weight = self.weight.reshape(self.out_channels, -1)
        out = torch.addmm(self.bias, gathered, weight.t()) if self.bias is not None else gathered @ weight.t()
        return x.replace_feature(out)


class SparseConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TRELLIS's inference path uses submanifold convolutions only")


class SparseInverseConv3d(SparseConv3d):
    pass
