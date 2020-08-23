import os
import torch
import argparse
import pickle 
from torchvision import transforms
from vocabulary import Vocabulary
from model import EncoderCNN, DecoderRNN, CategoryPredictor
from PIL import Image
import pandas as pd


class Img2Tags(object):
    def __init__(self, encoder_path, decoder_path, vocab_path,
                 embed_size=256, hidden_size=512, num_layers=1):
        # load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        # check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build models
        encoder = EncoderCNN(embed_size).eval()  # eval mode (batch-norm uses moving mean/variance)
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        # load model parameters
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        self.vocab = vocab
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def load_image(image, transform=None):
        image = image.convert('RGB').resize([224, 224], Image.LANCZOS)

        if transform is not None:
            image = transform(image).unsqueeze(0)

        return image

    def sample_process(self, image):
        device = self.device
        encoder = self.encoder
        decoder = self.decoder
        vocab = self.vocab
        # image pre-processing
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                )
            ]
        )
        # prepare an image
        image = self.load_image(image, transform)
        image_tensor = image.to(device)
        # Generate an caption from the image
        feature = encoder(image_tensor)
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
        return list(dict.fromkeys(sampled_caption))


def main(args):
    # load model
    ENCODER_PATH = args.encoder_path
    DECODER_PATH = args.decoder_path
    VOCAB_PATH = args.vocab_path
    EMBED_SIZE = args.embed_size
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYER = args.num_layers
    model = Img2Tags(ENCODER_PATH, DECODER_PATH, VOCAB_PATH, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYER)
    # preparing images and evaluation data if exist
    images_folder = args.images_folder
    label_csv = args.label_csv
    output_folder = args.output_folder
    df_real = None
    if not os.path.exists(label_csv):
        image_names = os.listdir(images_folder)
    else:
        df_real = pd.read_csv(label_csv)
        image_names = df_real['img_name'].tolist()
    num_images = len(image_names)
    # predict each image
    pred_records = []
    for i, img_name in enumerate(image_names):
        with open(os.path.join(images_folder, img_name), 'rb') as f:
            with Image.open(f) as img:
                img_tags = model.sample_process(img)
                img_tags_str = ",".join(img_tags)
                each_record = (img_name, img_tags_str)
                pred_records.append(each_record)
        if (i + 1) % 100 == 0:
            print("[{}/{}] images are processed.".format(i + 1, num_images))
    df_pred = pd.DataFrame(pred_records, columns=['img_name', 'pred_tags'])
    df_pred.to_csv(os.path.join(output_folder, 'images_prediction.csv'), index=False, header=True)
    if df_real is not None:
        df_eval = df_real.merge(df_pred, how='left', on='img_name')
        df_eval = df_eval.reindex(columns=['img_id', 'img_name', 'img_url', 'real_tags', 'pred_tags'])
        df_eval.to_csv(os.path.join(output_folder, 'images_evaluation.csv'), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, default='data/train', help='input image folder')
    parser.add_argument('--output_folder', type=str, default='source_data/', help='output folder')
    parser.add_argument('--label_csv', type=str, default='source_data/training_data_cleanup.csv', help='real csv')
    # Model parameters
    parser.add_argument('--encoder_path', type=str, default='models/encoder-final.ckpt', help='path trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-final.ckpt', help='path trained decoder')
    parser.add_argument('--vocab_path', type=str, default='models/vocab.pkl', help='path for vocabulary wrapper')
    # Model hyper parameters (should be same as parameters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
