from dataclasses import dataclass
from typing import List, Optional
from datetime import timedelta

@dataclass
class TranscriptSegment:
    start_time: float
    duration: float
    text: str
    confidence: float

@dataclass
class EditInstruction:
    type: str
    timestamp: float
    duration: float
    effect: Optional[str] = None

@dataclass
class AugmentedTranscript:
    segments: List[TranscriptSegment]
    edit_instructions: List[EditInstruction]
    style: str
    total_duration: float
