import time
import editdistance
import pandas as pd
import soundfile as sf
from pathlib import Path
from nemo.collections.asr.models import EncDecRNNTBPEModel

data_dir = Path('/root/datadrive/TEDxJP-10K_v1.1')
wav_dir = data_dir / 'test_dump'

# symbol.txtファイルを読み込んで、_SPECIALSに記号を追加
# https://github.com/reazon-research/ReazonSpeech/blob/master/pkg/_v1/src/data/symbol.txt
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



def calculate_cer(reference_texts, predicted_texts):
    total_errors = 0
    total_chars = 0
    
    for ref, pred in zip(reference_texts, predicted_texts):
        # Levenshtein距離を使ってエディット距離を計算する
        levenshtein_distance = editdistance.eval(ref, normalize(pred))
        total_errors += levenshtein_distance
        total_chars += len(ref)
    
    # CERを計算する
    cer = total_errors / total_chars
    return cer

def calc_duration(file_id):
        wav_path = wav_dir / f'{file_id}.wav'
        audio, sr = sf.read(wav_path)
        return len(audio) / sr



def main():
    df = pd.read_csv(data_dir / 'text', sep=' ', header=None, names=['id', 'text'])
    df['duration'] = df['id'].apply(calc_duration)
    id2text = df[['id', 'text']].set_index('id').to_dict()['text']
    paths2audio_files = [str(wav_dir / f'{file_id}.wav') for file_id in id2text.keys()]
    model = EncDecRNNTBPEModel.restore_from('/root/datadrive/reazonspeech-nemo-v2/reazonspeech-nemo-v2.nemo')
    
    tic = time.perf_counter()
    asr_text = model.transcribe(paths2audio_files)[0]
    toc = time.perf_counter() - tic
    
    gt_text = [id2text[file_id] for file_id in id2text.keys()]
    cer = calculate_cer(gt_text, asr_text)
    
    print(f'Elapsed time: {toc:.2f} seconds')
    print(f'RTF: {toc / df["duration"].sum():.2f}')
    print(f'CER: {cer:.2f}')
    
    with open('result_nemo.txt', 'w') as f:
        f.write(f'Elapsed time: {toc:.2f} seconds\n')
        f.write(f'RTF: {toc / df["duration"].sum():.2f}\n')
        f.write(f'CER: {cer:.2f}\n')
        
        for file_id, text in zip(id2text.keys(), asr_text):
            f.write(f'{file_id} {text}\n')
        
if __name__ == '__main__':
    main()