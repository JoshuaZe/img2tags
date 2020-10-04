import torch
import pickle
from torchvision import transforms
from img2tags.udf_transforms import SquarePad
from img2tags.model import EncoderCNN, DecoderRNN, CategoryPredictor
from PIL import Image

general_category_dict = {
    1: 'upper-body',
    2: 'lower-body',
    3: 'full-body'
}


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
        # general_categ_predictor = self.general_categ_predictor
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
        # out_general_category = general_categ_predictor(feature)
        # out_scores, out_indices = torch.sort(out_general_category, descending=True)
        # out_percents = torch.nn.functional.softmax(out_scores, dim=1)
        # general_category_id = out_indices[0][0].item() + 1
        # general_category_score = out_percents[0][0].item()
        # predicted_general_category = dict(
        #     category_id=general_category_id,
        #     category_name=general_category_dict.get(general_category_id),
        #     score=general_category_score
        # )
        # Generate an caption from the image
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids.flatten().cpu().numpy()
        # convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word.get(word_id, '<unk>')
            if word not in ['<start>', '<unk>', '<end>']:
                sampled_caption.append(word)
            if word == '<end>':
                break
        return list(set(sampled_caption))


if __name__ == '__main__':
    import os
    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    PROJ_DIR = os.path.abspath(os.path.join(APP_DIR, '..'))
    ENCODER_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/encoder-final.ckpt'))
    DECODER_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/decoder-final.ckpt'))
    VOCAB_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/vocab.pkl'))
    GENERAL_PREDICTOR_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/categ_predictor-final.ckpt'))
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    NUM_LAYER = 1
    NUM_GENERAL_CATEGORIES = 3
    model = Img2Tags(
        ENCODER_PATH, DECODER_PATH, VOCAB_PATH, GENERAL_PREDICTOR_PATH,
        EMBED_SIZE, HIDDEN_SIZE, NUM_LAYER, NUM_GENERAL_CATEGORIES
    )
    image = Image.open(
        '/Users/zezzhang/Workspace/img2tags_serving/image_prediction/data/A/train/image/id_00000001_855.jpg'
    )
    print(model.sample_process(image))
