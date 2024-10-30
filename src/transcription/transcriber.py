from typing import Dict, List
import logging
from openai import OpenAI
from .models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def transcribe(self, audio_file_path: str) -> List[TranscriptSegment]:
        try:
            logger.info(f"Starting transcription for: {audio_file_path}")
            
            with open(audio_file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            segments = []
            for segment in response.segments:
                segments.append(
                    TranscriptSegment(
                        start_time=segment.start,
                        duration=segment.end - segment.start,
                        text=segment.text,
                        confidence=segment.confidence
                    )
                )
            
            logger.info("Transcription completed successfully")
            return segments
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
