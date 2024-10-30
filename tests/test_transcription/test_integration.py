import pytest 
import os
from src.transcription.transcriber import Transcriber
from src.transcription.timestamp import TimestampHandler
from src.transcription.augmentor import TranscriptAugmentor

def test_full_transcription_pipeline(sample_segments):
    # Initialize components
    timestamp_handler = TimestampHandler()
    augmentor = TranscriptAugmentor(style="Fast & Energized")
    
    # Align timestamps
    aligned_segments = timestamp_handler.align_timestamps(sample_segments)
    
    # Augment transcript with editing instructions
    augmented_transcript = augmentor.augment(aligned_segments)
    
    # Assertions
    assert len(augmented_transcript.segments) == len(sample_segments)
    assert len(augmented_transcript.edit_instructions) == len(sample_segments)
    assert augmented_transcript.style == "Fast & Energized"
    
    # Check timing alignment
    assert augmented_transcript.segments[0].start_time == 0.0
    assert augmented_transcript.segments[1].start_time == sample_segments[0].duration
