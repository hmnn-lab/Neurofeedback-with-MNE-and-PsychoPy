import numpy as np
import mne
import mne_lsl
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import resolve_streams
import time
import os

def record_eeg_stream(stream_name, duration, filename="baseline.fif"):
    if not isinstance(stream_name, str) or not stream_name:
        raise ValueError("stream_name must be a non-empty string.")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("duration must be a positive number.")
    if not isinstance(filename, str) or not filename.endswith('.fif'):
        raise ValueError("filename must be a string ending with '.fif'.")

    print("Available LSL streams:")
    streams = resolve_streams(timeout=2.0)
    for stream in streams:
        print(f" - {stream.name} (type: {stream.stype})")

    try:
        stream = StreamLSL(bufsize=1.0, name=stream_name, stype='EEG')
        stream.connect(acquisition_delay=0.004, timeout=10.0)
    except Exception as e:
        raise ValueError(f"Failed to connect to stream {stream_name}: {e}")

    info = stream.info
    sfreq = info['sfreq']
    ch_names = info['ch_names']
    if not ch_names or not sfreq:
        stream.disconnect()
        raise ValueError("Could not retrieve channel names or sampling frequency from the stream.")

    samples = []
    print(f"Recording EEG data from {stream_name} for {duration} seconds...")
    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            data, _ = stream.get_data()
            if data.size > 0:
                samples.append(data)
            time.sleep(1)
    finally:
        stream.disconnect()
    print("Recording complete.")

    if not samples:
        raise ValueError("No data recorded from the stream.")

    data = np.concatenate(samples, axis=1)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # Save to incremented filename in subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_folder = os.path.join(script_dir, "baseline_recordings")
    os.makedirs(baseline_folder, exist_ok=True)

    index = 1
    while os.path.exists(os.path.join(baseline_folder, f"baseline_{index:03d}.fif")):
        index += 1
    filename = f"baseline_{index:03d}.fif"
    filepath = os.path.join(baseline_folder, filename)
    raw.save(filepath, overwrite=True)

    print(f"EEG data saved to {filepath}")
    return filepath
