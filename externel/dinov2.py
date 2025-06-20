from torchvision import transforms as T
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


class Dinov2Runner:
    def __init__(self, device='cuda') -> None:
        self.device = device
        dinov2_vits14 = torch.hub.load(
            './externel/dinov2', 'dinov2_vits14', source='local').half().to(self.device)
        dinov2_vits14.eval()
        dinov2_vits14.to(self.device)
        self.model = dinov2_vits14

        self.transform = T.Compose([
            # T.Resize(
            # size = smaller_edge_size, interpolation = T.InterpolationMode.BICUBIC, antialias = True),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(
                0.229, 0.224, 0.225)),  # imagenet defaults
        ])
        self.extract_features(np.zeros((500, 500, 3), dtype=np.uint8))

    @torch.inference_mode()
    def extract_features(self, rgb_image_numpy):
        start = time.time()
        image_size = rgb_image_numpy.shape[:2]
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)

        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - \
            height % self.model.patch_size  # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size,
                     cropped_width // self.model.patch_size)
        image_batch = image_tensor[None].half().to(self.device)
        tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        # 1, 384 dim, H patch, W patch
        tokens = tokens.cpu().reshape([*grid_size, -1]).permute(2, 0, 1)[None]

        print(f'dinov2 feature: {time.time()-start}s')
        return {
            'tokens': tokens.cpu(),
            'grid_size': grid_size,
            'patch_size': self.model.patch_size,
        }


def nms(a, maxn=5, r=2):
    n, m = a.shape
    topk = np.partition(a.reshape(-1), 100)
    topk = topk[100]
    cand = [(a[i, j], np.array([i, j]))
            for i in range(n) for j in range(m) if a[i, j] < topk]
    cand.sort(key=lambda x: x[0])
    # print(len(cand))
    ret = []
    for i, c in enumerate(cand):
        keep = True
        for x in cand[:i-1]:
            if np.linalg.norm(x[1]-c[1]) <= r:
                keep = False
                break
        if keep:
            ret.append(c)
            if len(ret) >= maxn:
                break
    return ret


if __name__ == '__main__':
    dinov2 = Dinov2Runner()
