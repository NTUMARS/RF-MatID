import torch
import torch.nn as nn
from transformers import ConvNextConfig, ConvNextModel


class RF_MatID(nn.Module):
    def __init__(self, seq_length, d_model, drop_rate, num_classes):
        super(RF_MatID, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model)
        self.linear_projection = LinearProjection(d_model)
        self.spatial_feature_extractor = SpatialFeatureExtractor(seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=256, dropout=drop_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pool = nn.AdaptiveAvgPool1d(12)
        self.norm = nn.LayerNorm(d_model)
        expand_size = ((384 // d_model) * 4) + 12
        self.cls_head = nn.Sequential(
            # nn.Linear(d_model * 2 * expand_size, 512),
            nn.Linear(d_model * expand_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Get positional encoding
        freq_position = x[0, :, 0]  # shape [N]
        pos_encoding = self.positional_encoding(freq_position)
        # Project dual channel complex data to higher dimension
        raw_data = x[:, :, 1:]  # shape [B, N, 2]
        projected_feature = self.linear_projection(raw_data)
        feature = pos_encoding + projected_feature # shape [B, N, d_model]
        # Pass through convnext
        spatial_feature = self.spatial_feature_extractor(feature) # shape [B, 384, 4]
        spatial_feature = spatial_feature.permute(0, 2, 1).contiguous()  # shape [B, 4, 384]
        spatial_feature = spatial_feature.view(spatial_feature.size(0), -1, self.d_model)  # shape [B, 4*384/d_model, d_model]
        temporal_feature = self.transformer_encoder(feature) # shape [B, N, d_model]
        temporal_feature = self.pool(temporal_feature.permute(0, 2, 1)).permute(0, 2, 1)  # shape [B, 12, d_model]
        fused_feature = torch.cat([spatial_feature, temporal_feature], dim=1)  # [B, ((384 // d_model) * 4) + 12, d_model]
        fused_feature = self.norm(fused_feature)
        fused_feature = fused_feature.view(fused_feature.size(0), -1)
        output = self.cls_head(fused_feature)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, min_freq=4., max_freq=43.5):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.min_freq = min_freq
        self.max_freq = max_freq

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, freq_position):
        # freq_position: [N]
        # normalize to [0, max_len - 1]
        position = (freq_position - self.min_freq) / (self.max_freq - self.min_freq) * (self.max_len - 1)
        position = position.unsqueeze(1)  # shape [N, 1]
        # ensure encoding uses same device
        encoding = torch.zeros(len(freq_position), self.d_model, device=freq_position.device)
        encoding[:, 0::2] = torch.sin(position * self.div_term)
        encoding[:, 1::2] = torch.cos(position * self.div_term)
        # encoding: [N, d_model]
        return encoding.unsqueeze(0)  # [1, N, d_model]


class LinearProjection(nn.Module):
    def __init__(self, d_model, in_features=2):
        super(LinearProjection, self).__init__()
        self.linear_projector = nn.Sequential(
            nn.Linear(in_features, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

    def forward(self, x):
        # x: [batch, N, 2]
        x = self.linear_projector(x)
        # x: [batch, N, d_model]
        return x
    

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, seq_length, d_model):
        super(SpatialFeatureExtractor, self).__init__()
        convnext_config = ConvNextConfig(
            num_channels = d_model,
            num_stages=3,
            image_size=32
        )
        convnext = ConvNextModel(convnext_config)
        convnext.layernorm = torch.nn.LayerNorm(384, eps=1e-12, elementwise_affine=True)
        self.convnext = convnext
        self.mapping = nn.Linear(seq_length, 1024)
        self.d_model = d_model

    def forward(self, x):
        # x: [B, seq_length, d_model]
        x = x.permute(0, 2, 1)  # [B, d_model, seq_length]
        x = self.mapping(x)  # [B, d_model, 1024]
        x = x.view(x.size(0), self.d_model, 32, 32)  # [B, d_model, 32, 32]
        output = self.convnext(x)
        output = output.last_hidden_state  # [B, 384, 2, 2]
        output = output.view(output.size(0), output.size(1), -1)  # [B, 384, 4]
        return output