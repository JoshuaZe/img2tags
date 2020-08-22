import os
import argparse
import pickle
from collections import Counter
from vocabulary import Vocabulary
from PIL import Image
import pandas as pd


def safe_div(x, y):
    if y == 0:
        return 0
    return x / y


def main(args):
    # load model and evaluation csv
    VOCAB_PATH = args.vocab_path
    evaluation_csv = args.evaluation_csv
    output_folder = args.output_folder
    df_eval = pd.read_csv(evaluation_csv)
    # instances level evaluation
    df_eval['precision'] = df_eval.apply(
        lambda r:
            len(set(str(r['real_tags']).split(',')) & set(str(r['pred_tags']).split(','))) / len(str(r['pred_tags']).split(','))
        ,
        axis=1
    )
    df_eval['recall'] = df_eval.apply(
        lambda r:
            len(set(str(r['real_tags']).split(',')) & set(str(r['pred_tags']).split(','))) / len(str(r['real_tags']).split(','))
        ,
        axis=1
    )
    print("In {} Samples, Average Precision is {} and Average Recall is {}.".format(
        len(df_eval.index),
        df_eval['precision'].mean(),
        df_eval['recall'].mean()
    ))
    df_eval.to_csv(os.path.join(output_folder, 'instances_evaluation.csv'), index=False, header=True)
    # tags level evaluation
    # counting
    counter_real = Counter()
    counter_pred = Counter()
    counter_correct = Counter()
    for index, row in df_eval.iterrows():
        row_real_tags = str(row['real_tags']).split(',')
        counter_real.update(row_real_tags)
        row_pred_tags = str(row['pred_tags']).split(',')
        counter_pred.update(row_pred_tags)
        row_correct_tags = list(set(row_real_tags) & set(row_pred_tags))
        counter_correct.update(row_correct_tags)
    # vocabulary
    tokens_record = []
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    for token in vocab.word2idx.keys():
        if token not in ['<pad>', '<start>', '<end>', '<unk>']:
            cnt_in_real = counter_real.get(token, 0)
            cnt_in_pred = counter_pred.get(token, 0)
            cnt_correct = counter_correct.get(token, 0)
            token_precision = safe_div(cnt_correct, cnt_in_pred)
            token_recall = safe_div(cnt_correct, cnt_in_real)
            each_record = (token, cnt_in_real, cnt_in_pred, cnt_correct, token_precision, token_recall)
            tokens_record.append(each_record)
    df_tags_eval = pd.DataFrame(
        tokens_record,
        columns=['tag', 'real_cnt', 'pred_cnt', 'correct_cnt', 'precision', 'recall']
    )
    print("In {} Tags, Average Precision is {} and Average Recall is {}.".format(
        len(df_tags_eval.index),
        df_tags_eval['precision'].mean(),
        df_tags_eval['recall'].mean()
    ))
    df_tags_eval.to_csv(os.path.join(output_folder, 'tags_evaluation.csv'), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, default='source_data/', help='output folder')
    parser.add_argument('--evaluation_csv', type=str, default='source_data/images_evaluation.csv', help='real csv')
    # Model parameters
    parser.add_argument('--vocab_path', type=str, default='models/vocab.pkl', help='path for vocabulary wrapper')
    args = parser.parse_args()
    main(args)
