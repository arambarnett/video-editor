import os
from dotenv import load_dotenv
from src.transcription.transcriber import Transcriber
from src.transcription.timestamp import TimestampHandler
from src.transcription.augmentor import TranscriptAugmentor

# Load environment variables
load_dotenv()

def process_video_transcript(video_path: str, style: str = "Moderate") -> dict:
    """
    Process a video through the complete transcription pipeline.
    """
    # Initialize components
    transcriber = Transcriber(api_key=os.getenv('OPENAI_API_KEY'))
    timestamp_handler = TimestampHandler()
    augmentor = TranscriptAugmentor(style=style)
    
    try:
        # Step 1: Transcribe video
        segments = transcriber.transcribe(video_path)
        
        # Step 2: Align timestamps
        aligned_segments = timestamp_handler.align_timestamps(segments)
        
        # Step 3: Augment with editing instructions
        augmented_transcript = augmentor.augment(aligned_segments)
        
        # Step 4: Format output
        return {
            "segments": [
                {
                    "timestamp": timestamp_handler.format_timestamp(seg.start_time),
                    "duration": seg.duration,
                    "text": seg.text,
                    "confidence": seg.confidence
                }
                for seg in augmented_transcript.segments
            ],
            "edit_instructions": [
                {
                    "timestamp": timestamp_handler.format_timestamp(instr.timestamp),
                    "type": instr.type,
                    "duration": instr.duration,
                    "effect": instr.effect
                }
                for instr in augmented_transcript.edit_instructions
            ],
            "style": augmented_transcript.style,
            "total_duration": augmented_transcript.total_duration
        }
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    video_path = "path/to/your/video.mp4"
    result = process_video_transcript(video_path, style="Fast & Energized")
    
    # Print results
    print("\nTranscript Segments:")
    for segment in result["segments"]:
        print(f"[{segment['timestamp']}] {segment['text']}")
    
    print("\nEdit Instructions:")
    for instruction in result["edit_instructions"]:
        print(f"[{instruction['timestamp']}] {instruction['type']} - {instruction['effect']}") 