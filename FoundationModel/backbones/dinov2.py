import torch
import torch.nn as nn
from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class DINOv2(nn.Module):

    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
    ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = torch.hub.load(
            '/home/sun/Desktop/LightWeight/knowledge_distillation/facebookresearch_dinov2_main',
            model_name, source='local')
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]  # [B, C] [16, 384]
        f = x[:, 1:]  # [B, C, H // 14, W // 14]
        print(f.shape)

        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)  # torch.Size([8, 768, 18, 18])
        print(f.shape)
        if self.return_token:
            return f, t
        # f = f.mean([-2, -1])
        return f


if __name__ == '__main__':
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    # Load the encoder model locally
    encoder = torch.hub.load(
        '/home/sun/Desktop/LightWeight/knowledge_distillation/facebookresearch_dinov2_main',
        'dinov2_vits14', source='local').cuda()

    patch_size = encoder.patch_size  # 14
    patch_h, patch_w = 20, 20  # Adjust based on GPU size
    background_threshold = 0.5

    # Load a single image
    image_path = "/home/sun/Desktop/CVGL/HKD_Version/0001_0001.jpg"  # Replace with your image file path
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


    # Preprocessing the image
    def transform(img):
        img = cv2.resize(img, (patch_h * patch_size, patch_w * patch_size)) / 255
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.24, 0.225])
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, 0).astype(np.float32)


    # Transform the image
    transformed_image = transform(image)

    x = torch.tensor(transformed_image).cuda()

    # Pass the input through the encoder
    features_dict = encoder.forward_features(x)
    features = features_dict["x_norm_patchtokens"].detach().cpu().numpy()

    patch_features = features.reshape(1 * patch_h * patch_w, -1)

    # Apply PCA and MinMax scaling
    pca = PCA(n_components=3)
    scaler = MinMaxScaler(clip=True)

    # First fit to separate background and foreground
    pca.fit(patch_features)
    pca_features = pca.transform(patch_features)

    # MinMax Scaling
    scaler.fit(pca_features)
    pca_features = scaler.transform(pca_features)

    # Threshold to separate background and foreground
    pca_background = pca_features[:, 0] > background_threshold
    pca_foreground = ~pca_background

    # Second fit for the object (foreground)
    pca.fit(patch_features[pca_foreground])
    pca_features_rem = pca.transform(patch_features[pca_foreground])

    scaler.fit(pca_features_rem)
    pca_features_rem = scaler.transform(pca_features_rem)

    # Map PCA features to RGB
    pca_features_rgb = np.zeros((1 * patch_h * patch_w, 3))
    pca_features_rgb[pca_background] = 0
    pca_features_rgb[pca_foreground] = pca_features_rem
    pca_features_rgb = pca_features_rgb.reshape(1, patch_h, patch_w, 3)

    # Plotting and saving the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot PCA components
    axes[0].imshow(pca_features_rgb[0])
    axes[0].set_title("PCA components")
    axes[0].axis("off")

    # Plot original image
    axes[1].imshow(image)
    axes[1].set_title("Original Image")
    axes[1].axis("off")

    plt.tight_layout()

    # Save the result
    output_path = "output_image.png"  # Replace with your desired output path
    plt.savefig(output_path)

    plt.show()




