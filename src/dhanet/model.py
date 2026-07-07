import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRegionConv3d(nn.Module):
    """Lightweight 3-D DRConv-style layer with voxel-wise expert mixing."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, experts=4):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                )
                for _ in range(experts)
            ]
        )
        self.gate = nn.Conv3d(in_channels, experts, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        stacked = torch.stack([conv(x) for conv in self.experts], dim=1)
        out = (stacked * weights.unsqueeze(2)).sum(dim=1)
        return self.bn(out)


class ConvMixerBlock3D(nn.Module):
    def __init__(self, dim, kernel_size=5, experts=4):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = DynamicRegionConv3d(dim, dim, kernel_size, padding=padding, groups=dim, experts=experts)
        self.pointwise = nn.Sequential(
            nn.Conv3d(dim, dim, 1, bias=False),
            nn.BatchNorm3d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = x + F.gelu(self.depthwise(x))
        return self.pointwise(x)


class DynamicConvMixer3D(nn.Module):
    def __init__(self, dim=64, depth=4, patch_size=4, kernel_size=5, experts=4, graph_shape=(4, 5, 4)):
        super().__init__()
        self.graph_shape = tuple(graph_shape)
        self.patch_embed = nn.Sequential(
            DynamicRegionConv3d(1, dim, patch_size, stride=patch_size, padding=0, experts=experts),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvMixerBlock3D(dim, kernel_size, experts) for _ in range(depth)])

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        return F.adaptive_avg_pool3d(x, self.graph_shape)


class PrototypeLayer(nn.Module):
    def __init__(self, channels, num_nodes=16, node_dim=128, graph_shape=(4, 5, 4)):
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_shape = tuple(graph_shape)
        self.assign = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(),
            nn.Linear(32, num_nodes),
        )
        self.node_embed = nn.Sequential(
            nn.LayerNorm(int(math.prod(self.graph_shape))),
            nn.Linear(int(math.prod(self.graph_shape)), node_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _peak_coords(x):
        b, c, d, h, w = x.shape
        flat_idx = x.flatten(2).argmax(dim=-1)
        z = flat_idx // (h * w)
        y = (flat_idx % (h * w)) // w
        x_coord = flat_idx % w
        coords = torch.stack([z, y, x_coord], dim=-1).float()
        denom = x.new_tensor([max(d - 1, 1), max(h - 1, 1), max(w - 1, 1)])
        return coords / denom * 2.0 - 1.0

    def forward(self, feature_map):
        b, c, d, h, w = feature_map.shape
        coords = self._peak_coords(feature_map)
        channel_mean = feature_map.flatten(2).mean(dim=-1, keepdim=True)
        channel_max = feature_map.flatten(2).amax(dim=-1, keepdim=True)
        assignment_input = torch.cat([coords, channel_mean, channel_max], dim=-1)
        scores = self.assign(assignment_input)
        weights = torch.softmax(scores.transpose(1, 2), dim=-1)
        node_maps = torch.bmm(weights, feature_map.flatten(2))
        node_features = self.node_embed(node_maps)
        node_coords = self._peak_coords(node_maps.reshape(b, self.num_nodes, d, h, w))
        return node_features, node_coords, node_maps, weights


class JointDynamicEdge(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.q = nn.Linear(node_dim, node_dim)
        self.k = nn.Linear(node_dim, node_dim)
        self.topology = nn.Linear(3, 1)
        self.delta = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_features, node_coords):
        q = self.q(node_features)
        k = self.k(node_features)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1)), dim=-1)
        diff = torch.tanh(node_coords.unsqueeze(2) - node_coords.unsqueeze(1))
        topo = self.topology(diff).squeeze(-1)
        adj = attn + self.delta * topo
        adj = torch.relu(adj)
        return 0.5 * (adj + adj.transpose(1, 2))


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye
        degree = adj.sum(dim=-1).clamp_min(1e-6)
        norm = degree.pow(-0.5).unsqueeze(-1) * adj * degree.pow(-0.5).unsqueeze(-2)
        return F.relu(torch.bmm(norm, self.proj(x)), inplace=True)


class DHANet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        channels=64,
        depth=4,
        node_dim=128,
        num_nodes=16,
        graph_shape=(4, 5, 4),
        patch_size=4,
        experts=4,
    ):
        super().__init__()
        self.backbone = DynamicConvMixer3D(channels, depth, patch_size, experts=experts, graph_shape=graph_shape)
        self.prototype = PrototypeLayer(channels, num_nodes, node_dim, graph_shape)
        self.edge = JointDynamicEdge(node_dim)
        self.gcn1 = GraphConv(node_dim, node_dim)
        self.gcn2 = GraphConv(node_dim, node_dim)
        self.backbone_classifier = nn.Linear(channels, num_classes)
        self.global_proj = nn.Linear(channels, node_dim)
        self.graph_classifier = nn.Sequential(
            nn.LayerNorm(node_dim * 2),
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, num_classes),
        )

    @staticmethod
    def _diversity_loss(node_features):
        x = F.normalize(node_features, dim=-1)
        sim = torch.matmul(x, x.transpose(1, 2))
        eye = torch.eye(sim.size(-1), device=sim.device).unsqueeze(0)
        return ((sim * (1.0 - eye)).clamp_min(0.0) ** 2).mean()

    @staticmethod
    def _hierarchy_loss(node_features, temperature=0.2):
        b, m, d = node_features.shape
        if m < 4:
            return node_features.new_tensor(0.0)
        fine = F.normalize(node_features, dim=-1)
        mid = F.normalize(F.adaptive_avg_pool1d(fine.transpose(1, 2), max(2, m // 2)).transpose(1, 2), dim=-1)
        coarse = F.normalize(F.adaptive_avg_pool1d(fine.transpose(1, 2), max(1, m // 4)).transpose(1, 2), dim=-1)
        child_to_mid = torch.matmul(fine, mid.transpose(1, 2)) / temperature
        target_mid = torch.arange(m, device=node_features.device) * mid.size(1) // m
        target_mid = target_mid.unsqueeze(0).expand(b, -1)
        loss_mid = F.cross_entropy(child_to_mid.reshape(b * m, -1), target_mid.reshape(-1))
        mid_to_coarse = torch.matmul(mid, coarse.transpose(1, 2)) / temperature
        target_coarse = torch.arange(mid.size(1), device=node_features.device) * coarse.size(1) // mid.size(1)
        target_coarse = target_coarse.unsqueeze(0).expand(b, -1)
        loss_coarse = F.cross_entropy(mid_to_coarse.reshape(b * mid.size(1), -1), target_coarse.reshape(-1))
        return 0.5 * (loss_mid + loss_coarse)

    def forward(self, image, debug_shapes=False):
        fb = self.backbone(image)
        global_vec = fb.flatten(2).mean(dim=-1)
        backbone_logits = self.backbone_classifier(global_vec)
        node_features, node_coords, node_maps, assignment = self.prototype(fb)
        adj = self.edge(node_features, node_coords)
        graph_features = self.gcn2(self.gcn1(node_features, adj), adj)
        graph_vec = graph_features.mean(dim=1)
        fused = torch.cat([self.global_proj(global_vec), graph_vec], dim=-1)
        graph_logits = self.graph_classifier(fused)
        if debug_shapes:
            print("image:", image.shape)
            print("fb:", fb.shape, "global_vec:", global_vec.shape)
            print("node_features:", node_features.shape, "node_coords:", node_coords.shape)
            print("adj:", adj.shape, "graph_features:", graph_features.shape)
            print("backbone_logits:", backbone_logits.shape, "graph_logits:", graph_logits.shape)
        return {
            "logits": graph_logits,
            "backbone_logits": backbone_logits,
            "graph_logits": graph_logits,
            "features": graph_vec,
            "node_features": node_features,
            "node_coords": node_coords,
            "node_maps": node_maps,
            "adjacency": adj,
            "assignment": assignment,
            "node_loss": self._diversity_loss(node_features),
            "edge_loss": self._hierarchy_loss(node_features),
        }
