from torch.utils.data import Dataset
from torchvision import transforms


class FIDImageDataset(Dataset):
    def __init__(self, images, postprocess):
        self.images = images
        self.postprocess = postprocess
        self.transform = transforms.Compose([transforms.Resize((128, 128))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.postprocess(image.unsqueeze(0))
        image = self.transform(image[0])
        dummy_label = 0
        return image, dummy_label
