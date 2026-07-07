import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = 0 if kernel_size == 2 else kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ConvUnit(in_channels, out_channels, 3),
            ConvUnit(out_channels, out_channels, 3),
        )

    def forward(self, x):
        return self.net(x)


class Backbone3D(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128, 256)):
        super().__init__()
        self.stem = ConvUnit(1, channels[0], kernel_size=2, stride=2)
        self.stage1 = ConvBlock(channels[0], channels[0])
        self.stage2 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(channels[0], channels[1]))
        self.stage3 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(channels[1], channels[2]))
        self.stage4 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(channels[2], channels[3]))
        self.stage5 = nn.Sequential(nn.MaxPool3d(2), ConvBlock(channels[3], channels[4]))

    def forward(self, x):
        x = self.stem(x)
        h1 = self.stage1(x)
        h2 = self.stage2(h1)
        h3 = self.stage3(h2)
        h4 = self.stage4(h3)
        h5 = self.stage5(h4)
        return h4, h5


class LogitsConstraintAttention(nn.Module):
    def __init__(self, channels=256, hidden=128, num_classes=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, high_map, target=None):
        tokens = high_map.flatten(2).transpose(1, 2)
        alpha = self.attn(tokens).squeeze(-1)
        weighted = tokens * alpha.unsqueeze(-1)
        pooled = weighted.mean(dim=1)
        image_logits = self.classifier(pooled)
        patch_logits = self.classifier(tokens)
        la_loss = high_map.new_tensor(0.0)
        if target is not None:
            patch_prob = torch.softmax(patch_logits, dim=-1)
            true_prob = patch_prob.gather(2, target.view(-1, 1, 1).expand(-1, patch_prob.size(1), 1)).squeeze(-1)
            la_loss = torch.sqrt(((true_prob - alpha) ** 2).sum(dim=1) + 1e-8).mean()
        return image_logits, tokens, alpha, la_loss


class GraphConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, adj):
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj = torch.maximum(adj, eye)
        degree = adj.sum(dim=-1).clamp_min(1.0)
        norm = degree.pow(-0.5).unsqueeze(-1) * adj * degree.pow(-0.5).unsqueeze(-2)
        return F.relu(torch.bmm(norm, self.proj(x)), inplace=True)


class GMFModule(nn.Module):
    def __init__(self, low_channels=128, high_channels=256, topk=16, threshold_high=0.4, threshold_low=0.0, num_classes=2):
        super().__init__()
        self.topk = topk
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.low_proj = nn.Linear(low_channels, high_channels)
        self.gcn_high = GraphConv(high_channels)
        self.gcn_low = GraphConv(high_channels)
        self.scale_attn = nn.MultiheadAttention(high_channels, num_heads=4, batch_first=True)
        self.agnn_high = GraphConv(high_channels)
        self.agnn_low = GraphConv(high_channels)
        self.classifier = nn.Linear(high_channels, num_classes)

    @staticmethod
    def _adjacency(x, threshold):
        sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)
        return (sim >= threshold).float()

    def _select_low_tokens(self, low_map, high_shape, top_idx):
        low_resized = F.adaptive_avg_pool3d(low_map, high_shape)
        low_tokens = low_resized.flatten(2).transpose(1, 2)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, low_tokens.size(-1))
        return low_tokens.gather(1, gather_idx)

    def forward(self, low_map, high_tokens, alpha, high_shape):
        k = min(self.topk, high_tokens.size(1))
        top_idx = alpha.topk(k=k, dim=1).indices
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, high_tokens.size(-1))
        high_selected = high_tokens.gather(1, gather_idx)

        low_selected = self._select_low_tokens(low_map, high_shape, top_idx)
        low_selected = self.low_proj(low_selected)

        adj_high = self._adjacency(high_selected, self.threshold_high)
        adj_low = self._adjacency(low_selected, self.threshold_low)
        h_high = self.gcn_high(high_selected, adj_high)
        h_low = self.gcn_low(low_selected, adj_low)

        scales = torch.stack([h_high, h_low], dim=2)
        flat = scales.reshape(scales.size(0) * scales.size(1), 2, scales.size(-1))
        attended = self.scale_attn(flat, flat, flat, need_weights=False)[0]
        fused = attended.reshape(scales.size(0), scales.size(1), 2, scales.size(-1)).flatten(2)
        fused_high, fused_low = fused.chunk(2, dim=-1)
        out = 0.5 * (self.agnn_high(fused_high, adj_high) + self.agnn_low(fused_low, adj_low))
        graph_logits = self.classifier(out.mean(dim=1))
        return graph_logits, out, top_idx


class LAGMF(nn.Module):
    def __init__(self, num_classes=2, topk=16):
        super().__init__()
        self.backbone = Backbone3D()
        self.la = LogitsConstraintAttention(256, 128, num_classes)
        self.gmf = GMFModule(128, 256, topk=topk, num_classes=num_classes)

    def forward(self, image, target=None, debug_shapes=False):
        low_map, high_map = self.backbone(image)
        image_logits, high_tokens, alpha, la_loss = self.la(high_map, target)
        graph_logits, graph_features, top_idx = self.gmf(low_map, high_tokens, alpha, high_map.shape[2:])
        logits = 0.5 * (image_logits + graph_logits)
        if debug_shapes:
            print("image:", image.shape)
            print("low_map:", low_map.shape, "high_map:", high_map.shape)
            print("high_tokens:", high_tokens.shape, "alpha:", alpha.shape)
            print("image_logits:", image_logits.shape, "graph_logits:", graph_logits.shape)
            print("graph_features:", graph_features.shape, "logits:", logits.shape)
        return {
            "logits": logits,
            "image_logits": image_logits,
            "graph_logits": graph_logits,
            "attention": alpha,
            "graph_features": graph_features,
            "top_idx": top_idx,
            "la_loss": la_loss,
        }
