import queue
import sys
import pyaudio
import numpy as np
from scipy.signal import find_peaks
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk

    def extract_features(self, chunk):
        data = np.frombuffer(chunk, dtype=np.int16)
        print(data)
        power = np.sqrt(np.mean(data**2))

        def autocorr(x):
            result = np.correlate(x, x, mode="full")
            return result[result.size // 2 :]

        def estimate_f0(data):
            corr = autocorr(data)
            peaks, _ = find_peaks(corr)
            if len(peaks) == 0:
                return 0
            peak = peaks[0]
            return self._rate / peak

        f0 = estimate_f0(data)

        mora = None
        return (power, f0, mora)


def log_features(power, f0, mora):
    logging.debug(f"Power: {power}, F0: {f0}, Mora: {mora}")


def main():
    import pyaudio

    pa = pyaudio.PyAudio()

    info = pa.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", pa.get_device_info_by_host_api_device_index(0, i).get('name'))

    
    with MicrophoneStream() as stream:
        for chunk in stream.generator():
            power, f0, mora = stream.extract_features(chunk)
            log_features(power, f0, mora)


if __name__ == "__main__":
    main()
