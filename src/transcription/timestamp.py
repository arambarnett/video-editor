from datetime import timedelta
from typing import List
from .models.transcript import TranscriptSegment

class TimestampHandler:
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        milliseconds = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    @staticmethod
    def align_timestamps(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Ensure timestamps are properly aligned and sequential"""
        current_time = 0.0
        aligned_segments = []
        
        for segment in segments:
            aligned_segment = TranscriptSegment(
                start_time=current_time,
                duration=segment.duration,
                text=segment.text,
                confidence=segment.confidence
            )
            aligned_segments.append(aligned_segment)
            current_time += segment.duration
            
        return aligned_segments
