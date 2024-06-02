import pandas as pd
import jiwer
from argparse import ArgumentParser

import MeCab

from dataclasses import dataclass


from num2words import num2words
import re
import neologdn

with open("symbol.txt", "r") as f:
    _SPECIALS = {ord(c.rstrip("\n")): "" for c in f.readlines()}


def normalize(utt_txt):
    """Normalize text.
    Use for Japanese text.
    Args:
        utt_txt: String of Japanese text.
    Returns:
        utt_txt: Normalized
    """
    # trim non-phonatory symbols in the text
    utt_txt = utt_txt.translate(_SPECIALS)
    # convert UTF-16 latin chars to ASCII
    utt_txt = neologdn.normalize(utt_txt)
    # replace some special characters
    utt_txt = utt_txt.replace('"', "").replace(",", "")
    # replace all the numbers
    numbers = re.findall(r"\d+\.?\d*", utt_txt)
    transcribed_numbers = [num2words(item, lang="ja") for item in numbers]
    for nr in range(len(numbers)):
        old_nr = numbers[nr]
        new_nr = transcribed_numbers[nr]
        utt_txt = utt_txt.replace(old_nr, new_nr, 1)
    return utt_txt


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
    parser.add_argument("input", type=str, help="Path to the result csv file")
    args = parser.parse_args()
    df = pd.read_csv(args.input)

    mecab = MeCab.Tagger("-Owakati")
    total_word_errors = TotalErrors()
    total_char_errors = TotalErrors()

    for row in df.itertuples():
        gt_text = normalize(row.gt_text)
        asr_text = normalize(row.pred_text)

        gt_words = mecab.parse(gt_text)
        asr_words = mecab.parse(asr_text)
        word_errors = jiwer.process_words(gt_words, asr_words)
        total_word_errors.deletions += word_errors.deletions
        total_word_errors.insertions += word_errors.insertions
        total_word_errors.substitutions += word_errors.substitutions
        distance = (
            word_errors.deletions + word_errors.insertions + word_errors.substitutions
        )
        total_word_errors.distance += distance
        total_word_errors.length += len(gt_words.split())

        char_errors = jiwer.process_characters(gt_text, asr_text)
        total_char_errors.deletions += char_errors.deletions
        total_char_errors.insertions += char_errors.insertions
        total_char_errors.substitutions += char_errors.substitutions
        distance = (
            char_errors.deletions + char_errors.insertions + char_errors.substitutions
        )
        total_char_errors.distance += distance
        total_char_errors.length += len(gt_text)

    print("----------- Word Error Rate -----------")
    print(f"WER: {total_word_errors.error_rate:.3f}")
    print("deletions:", total_word_errors.deletions)
    print("insertions:", total_word_errors.insertions)
    print("substitutions:", total_word_errors.substitutions)

    print("---------- Character Error Rate ----------")
    print(f"CER: {total_char_errors.error_rate:.3f}")
    print("deletions:", total_char_errors.deletions)
    print("insertions:", total_char_errors.insertions)
    print("substitutions:", total_char_errors.substitutions)


if __name__ == "__main__":
    main()
