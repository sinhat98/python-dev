from transformers import pipeline
import time
import pandas as pd
import soundfile as sf
from pathlib import Path
import time
from tqdm import tqdm
import torch

data_dir = Path('/root/datadrive/TEDxJP-10K_v1.1')
wav_dir = data_dir / 'test_dump'

def calc_duration(file_id):
        wav_path = wav_dir / f'{file_id}.wav'
        audio, sr = sf.read(wav_path)
        return len(audio) / sr


def main():
    device = "cuda"

    model_id = "kotoba-tech/kotoba-whisper-v1.0"
    torch_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model_kwargs = {"attn_implementation": "sdpa"} if device == 'cuda' else {}
    model = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs
    )
    generate_kwargs = {"language": "japanese", "task": "transcribe"}
    
    df = pd.read_csv(data_dir / 'text', sep=' ', header=None, names=['id', 'text'])
    df['duration'] = df['id'].apply(calc_duration)
    id2text = df[['id', 'text']].set_index('id').to_dict()['text']
    paths2audio_files = [str(wav_dir / f'{file_id}.wav') for file_id in id2text.keys()]
    results = []
    tic = time.perf_counter()
    for path in tqdm(paths2audio_files):
        
        ret = model(path, generate_kwargs=generate_kwargs)
        results.append({
                'gt_text': id2text[Path(path).stem],
                'pred_text': ret['text'],
            }
        )
    toc = time.perf_counter() - tic
    pd.DataFrame(results).to_csv('transcribe_results_kotobawhisper.csv', index=False)
    print(f'Elapsed time: {toc:.2f} seconds')
    print(f'RTF: {toc / df["duration"].sum():.2f}')
    
if __name__ == '__main__':
    main()