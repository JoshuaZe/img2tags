import os
import torch
import argparse
import pickle 
from torchvision import transforms
from img2tags.udf_transforms import SquarePad
from model import EncoderCNN, DecoderRNN, CategoryPredictor
from PIL import Image
import pandas as pd


class Img2Tags(object):
    def __init__(self, encoder_path, decoder_path, vocab_path, general_predictor_path,
                 embed_size=256, hidden_size=512, num_layers=1,
                 num_general_categories=3):
        # load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        # check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build models
        encoder = EncoderCNN(embed_size).eval()  # eval mode (batch-norm uses moving mean/variance)
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).eval()
        general_categ_predictor = CategoryPredictor(embed_size, num_general_categories).eval()

        encoder = encoder.to(device)
        decoder = decoder.to(device)
        general_categ_predictor = general_categ_predictor.to(device)

        # load model parameters
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        general_categ_predictor.load_state_dict(torch.load(general_predictor_path, map_location=device))

        self.device = device
        self.vocab = vocab
        self.encoder = encoder
        self.decoder = decoder
        self.general_categ_predictor = general_categ_predictor

    @staticmethod
    def load_image(image, bbox=None, transform=None):
        image = image.convert('RGB')

        if bbox is not None:
            [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, bbox should be [xmin, ymin, xmax, ymax]
            if bbox_h > 0 or bbox_w > 0:
                image = image.crop([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h])

        if transform is not None:
            image = transform(image).unsqueeze(0)

        return image

    def sample_process(self, image, bbox=None):
        device = self.device
        vocab = self.vocab
        encoder = self.encoder
        decoder = self.decoder
        general_categ_predictor = self.general_categ_predictor
        # image pre-processing
        transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            ]
        )
        # prepare an image
        image = self.load_image(image, bbox, transform)
        image_tensor = image.to(device)
        # Encoding
        feature = encoder(image_tensor)
        # General Category Prediction
        out_general_category = general_categ_predictor(feature)
        out_scores, out_indices = torch.sort(out_general_category, descending=True)
        out_percents = torch.nn.functional.softmax(out_scores, dim=1)
        predicted_general_category = dict(
            category_id=out_indices[0][0].item() + 1,
            score=out_percents[0][0].item()
        )
        # Generate an caption from the image
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids.flatten().cpu().numpy()
        # convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word not in ['<start>', '<unk>', '<end>']:
                sampled_caption.append(word)
            if word == '<end>':
                break
        return sampled_caption, predicted_general_category


def main(args):
    # load model
    ENCODER_PATH = args.encoder_path
    DECODER_PATH = args.decoder_path
    VOCAB_PATH = args.vocab_path
    GENERAL_PREDICTOR_PATH = args.general_predictor_path
    EMBED_SIZE = args.embed_size
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYER = args.num_layers
    NUM_GENERAL_CATEGORIES = args.num_general_categories
    model = Img2Tags(
        ENCODER_PATH, DECODER_PATH, VOCAB_PATH, GENERAL_PREDICTOR_PATH,
        EMBED_SIZE, HIDDEN_SIZE, NUM_LAYER, NUM_GENERAL_CATEGORIES
    )
    # preparing images and evaluation data if exist
    image_dir = args.image_dir
    image_meta_path = args.image_meta_path
    annotation_path = args.annotation_path
    output_folder = args.output_folder
    df_real = None
    if not os.path.exists(annotation_path):
        # assuming each image only has one object
        image_names = os.listdir(image_dir)
        obj_index = [0] * len(image_names)
        obj_bbox = [None] * len(image_names)
        lst_image_obj = list(zip(image_names, obj_index, image_names, obj_bbox))
    else:
        df_image = pd.read_csv(image_meta_path, keep_default_na=False)
        df_annotation_real = pd.read_csv(annotation_path, keep_default_na=False).head(10)
        df_real = df_annotation_real.merge(df_image, how='left', on='img_id')
        lst_image_obj = list(df_real[[
            'img_id', 'obj_id',
            'img_name', 'bbox'
        ]].to_records(index=False))
    num_objects = len(lst_image_obj)
    # predict each image object
    pred_records = []
    for i, (img_id, obj_id, img_name, bbox_str) in enumerate(lst_image_obj):
        with open(os.path.join(image_dir, img_name), 'rb') as f:
            with Image.open(f) as img:
                bbox = None
                if bbox_str is not None:
                    bbox = tuple(map(float, bbox_str.split(",")))
                img_tags, img_general_categ = model.sample_process(img, bbox)
                img_tags_str = ",".join(img_tags)
                general_category_id = img_general_categ.get('category_id')
                general_category_score = img_general_categ.get('score')
                each_record = (img_id, obj_id, img_tags_str, general_category_id, general_category_score)
                pred_records.append(each_record)
        if (i + 1) % 100 == 0:
            print("[{}/{}] images/objects are processed.".format(i + 1, num_objects))
    df_pred = pd.DataFrame(pred_records, columns=[
        'img_id', 'obj_id', 'pred_tags', 'pred_general_category', 'score_general_category'
    ])
    df_pred.to_csv(os.path.join(output_folder, 'image_objects_prediction.csv'), index=False, header=True)
    if df_real is not None:
        df_eval = df_real.merge(df_pred, how='left', on=['img_id', 'obj_id'])
        df_eval = df_eval.reindex(columns=[
            'img_id', 'obj_id',
            'attribute_tags', 'pred_tags',
            'general_category_id', 'pred_general_category', 'score_general_category'
        ])
        df_eval.to_csv(os.path.join(output_folder, 'image_objects_prediction.csv'), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/A/train/image', help='path for images')
    parser.add_argument('--image_meta_path', type=str, default='data/A/train/img_info.csv', help='path for image meta')
    parser.add_argument('--annotation_path', type=str, default='data/A/train/obj_info.csv', help='path for annotation')
    parser.add_argument('--output_folder', type=str, default='data/A/evaluation', help='output folder')
    # Model parameters
    parser.add_argument('--encoder_path', type=str, default='models/encoder-512.ckpt', help='path trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-512.ckpt', help='path trained decoder')
    parser.add_argument('--vocab_path', type=str, default='models/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--general_predictor_path', type=str, default='models/categ_predictor-512.ckpt', help='path trained predictor')
    # Model hyper parameters (should be same as parameters in train.py)
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--num_general_categories', type=int, default=3, help='number of general category')
    args = parser.parse_args()
    main(args)
