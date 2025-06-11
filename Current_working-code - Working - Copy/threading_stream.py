import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from mne_lsl.lsl import StreamInlet, resolve_streams
import time

class StreamThread(QThread):
    new_epoch = pyqtSignal(np.ndarray)  # emits epoch with shape (channels, samples)

    def __init__(self, lsl_stream_name, info, epoch_duration=1.0, step_duration=0.25, parent=None):
        super().__init__(parent)
        self.lsl_stream_name = lsl_stream_name
        self.info = info
        self.sfreq = info['sfreq']
        self.n_channels = len(self.info['ch_names'])
        self.epoch_duration = epoch_duration
        self.step_duration = step_duration  # Set to 0.25 for 4 Hz emission
        self.running = False
        self.inlet = None
        self.epoch_samples = int(self.sfreq * self.epoch_duration)
        self.step_samples = int(self.sfreq * self.step_duration)
        self.data_buffer = np.empty((0, self.n_channels))
        print(f"StreamThread initialized: "
              f"Epoch duration: {self.epoch_duration}s ({self.epoch_samples} samples), "
              f"Step duration: {self.step_duration}s ({self.step_samples} samples)")

    def run(self):
        try:
            streams = resolve_streams(timeout=5.0)
            found_stream = False
            for stream in streams:
                if stream.name == self.lsl_stream_name:
                    self.inlet = StreamInlet(stream)
                    found_stream = True
                    break
            if not found_stream:
                print(f"Stream {self.lsl_stream_name} not found!")
                return
        except Exception as e:
            print(f"Error resolving LSL streams: {e}")
            return

        self.running = True
        print(f"Waiting for initial buffer fill of {self.epoch_samples} samples...")
        while self.running and self.data_buffer.shape[0] < self.epoch_samples:
            samples, _ = self.inlet.pull_chunk(timeout=1.0)
            if samples is not None and len(samples) > 0:
                samples = np.array(samples)
                self.data_buffer = np.vstack([self.data_buffer, samples])

        if not self.running:
            return

        print("Initial buffer filled. Starting continuous processing with sliding window.")
        while self.running:
            if self.data_buffer.shape[0] >= self.epoch_samples:
                current_epoch = self.data_buffer[-self.epoch_samples:, :].T  # Shape: (channels, samples)
                self.new_epoch.emit(current_epoch)
            else:
                print("Warning: Buffer underflow, waiting for more samples...")

            # Pull new samples for the next step
            pulled_samples_count = 0
            required_samples = self.step_samples
            new_data_chunk = np.empty((0, self.n_channels))
            start_time = time.time()
            while self.running and pulled_samples_count < required_samples:
                samples, _ = self.inlet.pull_chunk(timeout=0.1)
                if samples is not None and len(samples) > 0:
                    samples_arr = np.array(samples)
                    new_data_chunk = np.vstack([new_data_chunk, samples_arr])
                    pulled_samples_count += samples_arr.shape[0]
                else:
                    time.sleep(0.005)  # Reduced sleep to maintain 0.25s pacing

            if not self.running:
                return

            # Update buffer
            if new_data_chunk.shape[0] > 0:
                self.data_buffer = np.vstack([self.data_buffer[new_data_chunk.shape[0]:, :], new_data_chunk])
            else:
                pass  # Buffer retains current state

            # Trim buffer to prevent excessive growth
            max_buffer_samples = self.epoch_samples + self.step_samples * 2
            if self.data_buffer.shape[0] > max_buffer_samples:
                self.data_buffer = self.data_buffer[-max_buffer_samples:, :]

            # Throttle to maintain ~0.25s interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.step_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        print("Stopping StreamThread...")
        self.running = False
        if self.inlet:
            self.inlet.close()
        self.wait()