import torch
import pickle
from torchvision import transforms
from img2tags.model import EncoderCNN, DecoderRNN
from PIL import Image


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


if __name__ == '__main__':
    import os
    APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    PROJ_DIR = os.path.abspath(os.path.join(APP_DIR, '..'))
    ENCODER_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/encoder-final.ckpt'))
    DECODER_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/decoder-final.ckpt'))
    VOCAB_PATH = os.path.abspath(os.path.join(PROJ_DIR, 'models/vocab.pkl'))
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYER = 1
    model = Img2Tags(ENCODER_PATH, DECODER_PATH, VOCAB_PATH, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYER)
    image = Image.open(
        '/Users/zezzhang/Workspace/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/train/z48682.jpg'
    )
    print(model.sample_process(image))
