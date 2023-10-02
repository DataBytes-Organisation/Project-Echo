#importing the files
from core import *
from my_io import *
from util import *
from exceptions import *
import auditok

def enhanced_event_detection(audio_filepath, sample_rate, energy_threshold=55, min_event_duration=0.2, max_silence_duration=0.3):
    # Use auditok's split method
    audio_regions = auditok.split(
        audio_filepath,
        sample_rate=sample_rate,
        min_dur=min_event_duration,
        max_dur=5,
        max_silence=max_silence_duration,
        energy_threshold=energy_threshold
    )

    # Extract start and end samples for each detected region
    events = [(int(r.meta.start * sample_rate), int(r.meta.end * sample_rate)) for r in audio_regions]

    print(f"Detected {len(events)} events.")

    return events
