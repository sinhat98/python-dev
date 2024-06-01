import time
import editdistance
import pandas as pd
import soundfile as sf
from pathlib import Path
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import time
from tqdm import tqdm

data_dir = Path('/root/datadrive/TEDxJP-10K_v1.1')
wav_dir = data_dir / 'test_dump'

def calc_duration(file_id):
        wav_path = wav_dir / f'{file_id}.wav'
        audio, sr = sf.read(wav_path)
        return len(audio) / sr

def main():
    # Load ReazonSpeech model from Hugging Face
    model = load_model()
    df = pd.read_csv(data_dir / 'text', sep=' ', header=None, names=['id', 'text'])
    df['duration'] = df['id'].apply(calc_duration)
    id2text = df[['id', 'text']].set_index('id').to_dict()['text']
    paths2audio_files = [str(wav_dir / f'{file_id}.wav') for file_id in id2text.keys()]
    results = []
    tic = time.perf_counter()
    for path in tqdm(paths2audio_files):
        
        # Read a local audio file
        audio = audio_from_path(path)
        # Recognize speech
        ret = transcribe(model, audio)

        results.append({
            'gt_text': id2text[Path(path).stem],
            'pred_text': ret.text,}
        )
    toc = time.perf_counter() - tic
    pd.DataFrame(results).to_csv('transcribe_results_nemo.csv', index=False)
    print(f'Elapsed time: {toc:.2f} seconds')
    print(f'RTF: {toc / df["duration"].sum():.2f}')
    
if __name__ == '__main__':
    main()
    
