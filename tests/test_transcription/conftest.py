import pytest
from src.transcription.models.transcript import TranscriptSegment

@pytest.fixture
def sample_segments():
    return [
        TranscriptSegment(
            start_time=0.0,
            duration=2.5,
            text="Hey guys, welcome to the video!",
            confidence=0.98
        ),
        TranscriptSegment(
            start_time=2.5,
            duration=3.0,
            text="Today we're going to explore something amazing.",
            confidence=0.95
        ),
        TranscriptSegment(
            start_time=5.5,
            duration=2.0,
            text="Let's get started!",
            confidence=0.99
        )
    ]
