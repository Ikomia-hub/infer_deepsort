import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from .model import Net
from torchreid import models


class Extractor(object):
    def __init__(self, reid_model_name, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model = self._load_model(reid_model_name)
        self.model.to(self.device)
        self.model.eval()

        self.size = (128, 256)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_model(self, name):
        if name == "original":
            model_path = os.path.join(os.path.dirname(__file__), "checkpoint/ckpt.t7")
            model = Net(reid=True)
            state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
            model.load_state_dict(state_dict)
            logger = logging.getLogger("root.tracker")
            logger.info("Loading weights from {}... Done!".format(model_path))
        else:
            model = models.build_model(name=name, num_classes=1000)

        return model

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
