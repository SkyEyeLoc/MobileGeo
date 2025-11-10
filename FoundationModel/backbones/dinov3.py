import torch
import torch.nn as nn
from modelscope import AutoModel


class DINOv3(nn.Module):
    def __init__(self, pretrained_model_name, num_channels):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name,
        )
        self.pretrained_model_name = pretrained_model_name
        self.num_channels = num_channels
        self.config = self.backbone.config

    def forward(self, pixel_values):
        # print(self.config)
        outputs = self.backbone(pixel_values=pixel_values)
        all_tokens = outputs.last_hidden_state  # pooler_output
        cls_token_features = all_tokens[:, 0]

        num_register_tokens = 4
        if 'convnext' not in self.pretrained_model_name:
            patch_token = all_tokens[:, 1 + num_register_tokens:]  # [16, 49, 768]
        else:
            patch_token = all_tokens[:, 1:]

        """
        dinov3-convnext-tiny-pretrain-lvd1689m
        50: 序列长度 (Sequence Length)。
        1 个 [CLS] Token（用于分类）+ 49 个 Patch Token。
        49 个 Patches: 这意味着模型将 224x224 的图像分成了 7x7 的网格（因为 7 * 7 = 49）。
        Patch Size: 要得到 7x7 的网格，Patch Size 必须是 32x32（因为 224 / 32 = 7）
        """
        B, L, C = all_tokens.shape
        num_patches = L - 1  # 49
        H = W = int(num_patches**0.5)  # H=7, W=7
        # print("patch_token:", patch_token.shape)
        patch_token = patch_token.reshape(B, H, W, C)
        patch_token_feature = patch_token.permute(0, 3, 1, 2).contiguous()  # [16, 768, 7, 7]
        return patch_token_feature, cls_token_features


if __name__ == '__main__':
    pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"  # dinov3-vitb16-pretrain-lvd1689m dinov3-convnext-tiny-pretrain-lvd1689m
    num_channels = 768

    base_model = DINOv3(pretrained_model_name,num_channels)
    dummy = torch.randn(16, 3, 224, 224)
    output = base_model(dummy)
