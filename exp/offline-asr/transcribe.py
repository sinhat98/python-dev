import time
import torch
from argparse import ArgumentParser

class SpeechToText:
    def __init__(self, model_name, device=None):
        """Initializer for SpeechToText class

        Args:
          model_name (str): Model name
          device (str): Specify "cuda" or "cpu"
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        self.model = self._init_model(model_name)
        
    def _init_model(self, name):
        if name == 'reazonspeech-nemo':
            from reazonspeech.nemo.asr import load_model
            return load_model(self.device)
        elif name == 'kotoba-whisper':
            from transformers import pipeline
            model_id = "kotoba-tech/kotoba-whisper-v1.0"
            torch_dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            model_kwargs = {"attn_implementation": "sdpa"} if self.device == 'cuda' else {}
            return pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=self.device,
                model_kwargs=model_kwargs
            )
        elif name == 'nue-asr':
            import nue_asr
            from functools import partial
            
            model = nue_asr.load_model("rinna/nue-asr")
            model = model.to(self.device)
            tokenizer = nue_asr.load_tokenizer("rinna/nue-asr")
            trascribe_fn = partial(nue_asr.transcribe, model=model, tokenizer=tokenizer)
            return trascribe_fn
            
    def transcribe(self, audio_path):
        """Transcribe the given audio file

        Args:
          audio_path (str): Path to the audio file

        Returns:
          tuple: Transcription text and elapsed time
        """
        tic = time.perf_counter()

        if self.model_name == 'reazonspeech-nemo':
            from reazonspeech.nemo.asr import transcribe, audio_from_path
            audio = audio_from_path(audio_path)
            asr_text = transcribe(self.model, audio).text
        elif self.model_name == 'kotoba-whisper':
            generate_kwargs = {"language": "japanese", "task": "transcribe"}
            asr_text = self.model(audio_path, generate_kwargs=generate_kwargs)['text']
        elif self.model_name == 'nue-asr':
            asr_text = self.model(audio_path)
        else:
            raise ValueError(f"Model {self.model_name} is not supported")
        
        toc = time.perf_counter() - tic
        return asr_text, toc
    
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='reazonspeech-nemo', help='Model name', choices=['reazonspeech-nemo', 'kotoba-whisper', 'nue-asr'])
    parser.add_argument('--audio_and_text', type=str, help='Path to the audio and text file')
    stt = SpeechToText('reazonspeech-nemo')
    