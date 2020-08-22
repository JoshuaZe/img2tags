import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from udf_transforms import SquarePad
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab=None, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.num_categories = len(self.coco.cats)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image instance, target).

        target is (category, color tags, attribute tags)
        """
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        [bbox_x, bbox_y, bbox_w, bbox_h] = coco.anns[ann_id]['bbox']
        category_id = coco.anns[ann_id]['category_id']
        color_tags = coco.anns[ann_id]['color_tags']
        attribute_tags = coco.anns[ann_id]['attribute_tags']

        # make image object
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, bbox should be [xmin, ymin, xmax, ymax]
        if bbox_h > 0 and bbox_w > 0:
            image = image.crop([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h])
        if self.transform is not None:
            image = self.transform(image)

        # make category
        label_category_id = torch.LongTensor([category_id - 1])
        label_category = torch.zeros(self.num_categories).scatter_(0, label_category_id, 1)

        # make attribute tags
        lst_attributes = list()
        if self.vocab is not None:
            # Convert caption (string) to word ids.
            lst_attributes.append(vocab('<start>'))
            lst_attributes.extend([vocab(token) for token in attribute_tags])
            lst_attributes.append(vocab('<end>'))
        label_attributes = torch.as_tensor(lst_attributes)

        return image, label_category, label_attributes

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets_attributes: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, labels_category, labels_attributes = zip(*batch)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge category one-hot
    targets_category = torch.stack(labels_category, 0)
    
    # Merge attribute tags (from tuple of 1D tensor to 2D tensor).
    lengths = [len(tags) for tags in labels_attributes]
    targets_attributes = torch.zeros(len(labels_attributes), max(lengths)).long()
    for i, tags in enumerate(labels_attributes):
        end = lengths[i]
        targets_attributes[i, :end] = tags[:end]
    return images, targets_category, targets_attributes, lengths


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    os.chdir("/Users/zezzhang/Workspace/img2tags_serving/image_prediction")
    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15, expand=False),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.ToTensor()
    ])

    data_loader = get_loader(
        'data/A/train/image',
        'data/A/train/train_coco.json', None,
        transform, 128,
        shuffle=True, num_workers=2
    )

    for images, targets_category, targets_attributes, lengths in data_loader:
        print(targets_category[0])
        print(targets_attributes[0])
        print(lengths[0])
        img = transforms.ToPILImage()(images[0])
        plt.imshow(img)
        plt.show()
