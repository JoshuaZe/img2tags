import re
import pandas as pd
import pickle
import argparse
from collections import Counter
from vocabulary import Vocabulary


def has_numbers(token):
    return any(char.isdigit() for char in token)


def build_vocab(annotation_path, threshold):
    """Build a simple vocabulary wrapper."""
    df_annotation = pd.read_csv(annotation_path, keep_default_na=False)
    counter = Counter()
    for _, each_annotation in df_annotation.iterrows():
        attribute_tags = each_annotation['attribute_tags']
        tokens = list(re.split('[,]', attribute_tags))
        if len(tokens) > 0:
            tokens = [token.strip() for token in tokens if not has_numbers(token)]
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # print(words)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(annotation_path=args.annotation_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str,
                        default='./data/A/train/obj_info.csv',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str,
                        default='./models/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=10,
                        help='minimum tag count threshold')
    args = parser.parse_args()
    main(args)
