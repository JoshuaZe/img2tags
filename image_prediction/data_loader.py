import os
import torch
import torchvision.transforms as transforms
from udf_transforms import SquarePad
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import pickle


class TagsAndGeneralCategoryDataset(data.Dataset):
    """User Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, image_meta_path, annotation_path, vocab=None, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            annotation_path: annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        # img_id,img_name,img_url,source
        self.df_image = pd.read_csv(image_meta_path, keep_default_na=False)
        # img_id,obj_id,item_id,bbox,category_id,category_name,color_tags,attribute_tags
        df_annotation = pd.read_csv(annotation_path, keep_default_na=False)
        df_annotation = df_annotation.query("attribute_tags != '' and general_category_id != ''")
        self.df_annotation = df_annotation
        self.num_categories = len(df_annotation.general_category_id.unique())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image instance, target).

        target is (category, attribute tags)
        """
        annotation = self.df_annotation.iloc[index, :]
        image_meta = self.df_image.query("img_id == '{}'".format(annotation.img_id)).iloc[0]

        # make image object
        image = Image.open(os.path.join(self.root, image_meta.img_name)).convert('RGB')
        bbox_x = 0
        bbox_y = 0
        bbox_w = 0
        bbox_h = 0
        if annotation.bbox != "":
            [bbox_x, bbox_y, bbox_w, bbox_h] = tuple(map(float, annotation.bbox.split(",")))
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, bbox should be [xmin, ymin, xmax, ymax]
        if bbox_h > 0 or bbox_w > 0:
            image = image.crop([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h])
        if self.transform is not None:
            image = self.transform(image)

        # make category
        category_id = annotation.general_category_id
        label_category = torch.LongTensor([category_id - 1]).squeeze()
        # label_category = torch.zeros(self.num_categories).scatter_(0, label_category_id, 1)

        # make attribute tags
        vocab = self.vocab
        attribute_tags = list(annotation.attribute_tags.split(","))
        lst_attributes = list()
        if vocab is not None:
            # Convert caption (string) to word ids.
            lst_attributes.append(vocab('<start>'))
            lst_attributes.extend([vocab(token) for token in attribute_tags])
            lst_attributes.append(vocab('<end>'))
        label_attributes = torch.as_tensor(lst_attributes)

        return image, label_category, label_attributes

    def __len__(self):
        return len(self.df_annotation.index)


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
        targets_category: torch tensor of shape (batch_size, num_categories).
        targets_attributes: torch tensor of shape (batch_size, padded_length).
        attributes_lengths: list; valid length for each padded caption.
    """
    batch.sort(key=lambda x: len(x[2]), reverse=True)
    images, labels_category, labels_attributes = zip(*batch)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge category one-hot
    targets_category = torch.stack(labels_category, 0)
    
    # Merge attribute tags (from tuple of 1D tensor to 2D tensor).
    attributes_lengths = torch.IntTensor([len(tags) for tags in labels_attributes])
    targets_attributes = torch.zeros(len(labels_attributes), max(attributes_lengths)).long()
    for i, tags in enumerate(labels_attributes):
        end = attributes_lengths[i]
        targets_attributes[i, :end] = tags[:end]
    return images, targets_category, targets_attributes, attributes_lengths


def get_loader(root, image_meta_path, annotation_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO dataset
    coco = TagsAndGeneralCategoryDataset(
        root=root,
        image_meta_path=image_meta_path,
        annotation_path=annotation_path,
        vocab=vocab, transform=transform
    )
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

    # Load vocabulary wrapper
    vocab_path = "models/vocab.pkl"
    vocab = None
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

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
        'data/A/train/img_info.csv',
        'data/A/train/obj_info.csv',
        vocab=vocab, transform=transform,
        batch_size=128, shuffle=True, num_workers=2
    )

    for images, targets_category, targets_attributes, attributes_lengths in data_loader:
        print(targets_category[0])
        print(targets_attributes[0])
        print(attributes_lengths)
        img = transforms.ToPILImage()(images[0])
        plt.imshow(img)
        plt.show()
        break
