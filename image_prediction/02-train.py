import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from udf_transforms import SquarePad
from model import EncoderCNN, DecoderRNN, CategoryPredictor, GPUDataParallel
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Image pre-processing, normalization for the pre-trained resnet
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15, expand=False),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 1), value=(1, 1, 1)),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])
    
    # Build data loader
    data_loader = get_loader(
        args.image_dir,
        args.image_meta_path,
        args.annotation_path,
        vocab=vocab, transform=transform,
        batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    # Build the models
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    categ_predictor = CategoryPredictor(args.embed_size, args.num_categories)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = GPUDataParallel(encoder)
        decoder = GPUDataParallel(decoder)
        categ_predictor = GPUDataParallel(categ_predictor)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    categ_predictor = categ_predictor.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params_encoder = list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    params_decoder = list(decoder.parameters())
    params_predictor = list(categ_predictor.parameters())
    params = params_encoder + params_decoder + params_predictor
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, categories, attributes, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            targets_category = categories.to(device)
            attributes = attributes.to(device)
            targets_attributes = pack_padded_sequence(attributes, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs_attributes = decoder(features, attributes, lengths)
            loss_attributes = criterion(outputs_attributes, targets_attributes)
            outputs_category = categ_predictor(features)
            loss_category = criterion(outputs_category, targets_category)
            encoder.zero_grad()
            decoder.zero_grad()
            categ_predictor.zero_grad()
            loss = loss_attributes + loss_category
            loss.backward()
            optimizer.step()

            # Print log info
            if (i + 1) % args.log_step == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Category Perplexity: {:5.4f}, Attributes Perplexity: {:5.4f}'.format(
                        epoch + 1, args.num_epochs,
                        i + 1, total_step,
                        np.exp(loss_category.item()), np.exp(loss_attributes.item())
                    )
                )

        # Save the model checkpoints
        if (epoch + 1) % args.save_step == 0:
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-{}.ckpt'.format(epoch+1)))
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-{}.ckpt'.format(epoch+1)))
            torch.save(categ_predictor.state_dict(), os.path.join(
                args.model_path, 'categ_predictor-{}.ckpt'.format(epoch+1)))

    # final save
    torch.save(encoder.state_dict(), os.path.join(
        args.model_path, 'encoder-final.ckpt'))
    torch.save(decoder.state_dict(), os.path.join(
        args.model_path, 'decoder-final.ckpt'))
    torch.save(categ_predictor.state_dict(), os.path.join(
        args.model_path, 'categ_predictor-final.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='models/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/A/train/image', help='path for images')
    parser.add_argument('--image_meta_path', type=str, default='data/A/train/img_info.csv', help='path for image meta')
    parser.add_argument('--annotation_path', type=str, default='data/A/train/obj_info.csv', help='path for annotation')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of LSTM hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM')
    parser.add_argument('--num_categories', type=int, default=3, help='number of Category Classes')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')
    args = parser.parse_args()
    print(args)
    main(args)

