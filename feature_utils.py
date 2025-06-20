import torch


def postprocess(features, device='cuda', upsample=True):
    assert 'model_name' in features, print('expect model_name in features')
    model_name = features['model_name']
    assert upsample == True or model_name == 'dinov2'
    if model_name == 'dift':
        img_features = features['tokens'].to(device)
        image_size = features['image_size']
        img_features = torch.nn.Upsample(
            size=image_size, mode='bilinear')(img_features)
        img_features = torch.nn.functional.normalize(img_features)
        img_features = img_features[0].permute(1, 2, 0)

    elif model_name == 'dinov2':
        tokens = features['tokens'].to(device)
        image_size = [x * features['patch_size']
                      for x in features['grid_size']]
        if upsample:
            tokens = torch.nn.Upsample(
                size=image_size, mode='bilinear')(tokens)
        tokens = torch.nn.functional.normalize(tokens)
        img_features = tokens[0].permute(1, 2, 0)

    elif model_name == 'sddino':
        desc = features['desc'].to(device)
        h, w = features['image_size']
        s = max(h, w)
        desc = desc.permute(0, 3, 1, 2)
        desc = torch.nn.Upsample(size=(s, s), mode='bilinear')(desc)
        desc = torch.nn.functional.normalize(desc)
        if h <= w:
            desc = desc[:, :, (w-h)//2:(w-h)//2+h, :]
        else:
            desc = desc[:, :, :, (h-w)//2:(h-w)//2+w]
        img_features = desc[0].permute(1, 2, 0)

    elif model_name == 'geoaware':
        desc = features['desc'].to(device)
        h, w = features['image_size']
        s = max(h, w)
        desc = torch.nn.Upsample(size=(s, s), mode='bilinear')(desc)
        desc = torch.nn.functional.normalize(desc)
        if h <= w:
            desc = desc[:, :, (w-h)//2:(w-h)//2+h, :]
        else:
            desc = desc[:, :, :, (h-w)//2:(h-w)//2+w]
        img_features = desc[0].permute(1, 2, 0)

    else:
        raise NotImplementedError

    return img_features
