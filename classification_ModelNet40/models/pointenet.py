# Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


class MAA(nn.Module):
    def __init__(self, in_channels, group_num, features_num):
        super(MAA, self).__init__()
        self.in_channels = in_channels
        self.group_num = group_num
        self.features_num = features_num
        self.alpha_list = []
        self.beta_list = []
        for i in range(self.features_num):
            self.alpha_list.append(nn.Parameter(torch.ones([1, self.in_channels, 1])))
            self.beta_list.append(nn.Parameter(torch.zeros([1, self.in_channels, 1])))
        self.alpha_list = nn.ParameterList(self.alpha_list)
        self.beta_list = nn.ParameterList(self.beta_list)

        self.linear = Linear1Layers(self.in_channels, self.in_channels, bias=False)

    def forward(self, features_list):
        assert len(features_list) == self.features_num
        for i in range(self.features_num):
            features_list[i] = self.alpha_list[i] * features_list[i] + self.beta_list[i]

        features_list = torch.stack(features_list).sum(dim=0)
        features_list = self.linear(features_list)
        return features_list


# Local Geometry Aggregation
class MLGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion, surface_points, group_num):
        super().__init__()
        self.surface_points = surface_points
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)

        self.Pooling = Pooling()

        self.norm_embedding = Linear1Layer(3, out_dim, bias=False)
        self.curv_embedding = Linear1Layer(3, out_dim, bias=False)
        self.MAA = MAA(out_dim, group_num, 3)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # Surface Normal and Curvature
        if self.surface_points is not None:
            est_normal, est_curvature = get_local_geo(knn_xyz[..., :self.surface_points, :])
            est_normal = self.norm_embedding(est_normal.permute(0, 2, 1))
            est_curvature = self.curv_embedding(est_curvature.permute(0, 2, 1))

        # Normalization
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = self.linear1(knn_x.reshape(B, -1, G * K)).reshape(B, -1, G, K)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        # Pooling
        knn_x_w = self.Pooling(knn_x_w)

        if self.surface_points is not None:
            knn_x_w = self.MAA([knn_x_w, est_normal, est_curvature])
        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x


# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class Linear1Layers(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layers, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels / 2),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels / 2)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels / 2), out_channels=in_channels,
                      kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


def get_local_geo(knn_xyz):
    # Surface Normal and Curvature
    centroid = knn_xyz.mean(dim=2, keepdim=True)
    matrix1 = torch.matmul(centroid.permute(0, 1, 3, 2), centroid)
    matrix2 = torch.matmul(knn_xyz.permute(0, 1, 3, 2), knn_xyz) / knn_xyz.shape[2]
    matrix = matrix1 - matrix2
    u, s, v = torch.svd(matrix)
    est_normal = v[:, :, :, 2]
    est_normal = est_normal / torch.norm(est_normal, p=2, dim=-1, keepdim=True)
    est_curvature = s + 1e-9
    est_curvature = est_curvature / est_curvature.sum(dim=-1, keepdim=True)

    return est_normal, est_curvature


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w


# Parametric Encoder
class MGE(nn.Module):
    def __init__(self, in_channels, input_points, num_stages, embed_dim, k_neighbors, k_neighbors_list, alpha,
                 beta, MLGA_block, dim_expansion):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.MLGA_list = nn.ModuleList()  # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.MLGA_list.append(MLGA(out_dim, self.alpha, self.beta, MLGA_block[i], dim_expansion[i],
                                       surface_points=k_neighbors_list[i], group_num=group_num))

    def forward(self, xyz, x):

        # Raw-point Embedding
        # pdb.set_trace()
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            x = self.MLGA_list[i](xyz, lc_x, knn_xyz, knn_x)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)
        return x


# Parametric Network for ModelNet40
class PointeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=40, input_points=1024, num_stages=4,
                 embed_dim=36, k_neighbors=40, k_neighbors_list=[40, 12, None, None],
                 beta=100, alpha=1000, MLGA_block=[1, 1, 1, 1], dim_expansion=[2, 2, 2, 1]):
        super().__init__()
        # Parametric Encoder
        self.MGE = MGE(in_channels, input_points, num_stages, embed_dim, k_neighbors,
                       k_neighbors_list, alpha, beta, MLGA_block, dim_expansion)
        self.out_channel = embed_dim
        for i in dim_expansion:
            self.out_channel *= i
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # xyz: point coordinates
        # x: point features, size: bs*3*1024
        xyz = x.permute(0, 2, 1).contiguous()  # bs*1024*3

        # Parametric Encoder
        x = self.MGE(xyz, x)  # bs*288

        # Classifier
        x = self.classifier(x)  # bs*40
        return x


if __name__ == '__main__':
    data_x = torch.rand(11, 3, 1024).cuda()
    model = PointeNet(input_points=1024, in_channels=3, num_classes=40).cuda()
    out = model(data_x)
    print(out.shape)
