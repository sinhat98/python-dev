import pandas as pd
import jiwer
from argparse import ArgumentParser

import MeCab

from dataclasses import dataclass

with open('symbol.txt', 'r') as f:
    _SPECIALS = {ord(c.rstrip("\n")): "" for c in f.readlines()}

_HAN2ZEN = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９")

def normalize(text):
    """Trim non-phonatory symbols in the text

    Args:
        text(str): A string to process

    Returns:
        A normalized string
    """
    text = text.replace('<sos/eos>', '')
    text = text.replace('<unk>', '')
    return text.translate(_SPECIALS).translate(_HAN2ZEN)


@dataclass
class TotalErrors:
    deletions: int = 0
    insertions: int = 0
    substitutions: int = 0
    distance: int = 0
    length: int = 0
    
    @property
    def error_rate(self):
        return self.distance / self.length


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the result csv file')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    
    mecab = MeCab.Tagger("-Owakati")
    total_word_errors = TotalErrors()
    total_char_errors = TotalErrors()
    
    for row in df.itertuples():
        gt_text = row.gt_text
        asr_text = normalize(row.pred_text)
        
        gt_words = mecab.parse(gt_text)
        asr_words = mecab.parse(asr_text)
        word_errors = jiwer.process_words(gt_words, asr_words)
        total_word_errors.deletions += word_errors.deletions
        total_word_errors.insertions += word_errors.insertions
        total_word_errors.substitutions += word_errors.substitutions
        distance = word_errors.deletions + word_errors.insertions + word_errors.substitutions
        total_word_errors.distance += distance
        total_word_errors.length += len(gt_words.split())
        
        char_errors = jiwer.process_characters(gt_text, asr_text)        
        total_char_errors.deletions += char_errors.deletions
        total_char_errors.insertions += char_errors.insertions
        total_char_errors.substitutions += char_errors.substitutions
        distance = char_errors.deletions + char_errors.insertions + char_errors.substitutions
        total_char_errors.distance += distance
        total_char_errors.length += len(gt_text)
    
    print('----------- Word Error Rate -----------')
    print(f'WER: {total_word_errors.error_rate:.3f}')
    print('deletions:', total_word_errors.deletions)
    print('insertions:', total_word_errors.insertions)
    print('substitutions:', total_word_errors.substitutions)
    
    print('---------- Character Error Rate ----------')
    print(f'CER: {total_char_errors.error_rate:.3f}')        
    print('deletions:', total_char_errors.deletions)
    print('insertions:', total_char_errors.insertions)
    print('substitutions:', total_char_errors.substitutions)
    

if __name__ == "__main__":
    main()