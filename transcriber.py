import os
import subprocess
import speech_recognition as sr
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account
from openai import OpenAI
from dotenv import load_dotenv
import time
import tempfile
import json
import re
import logging
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, concatenate_videoclips
import emoji
from editing_presets import PRESETS
import random
import traceback
from google.cloud import storage
from google.api_core import retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_credentials():
    try:
        # Use absolute path to credentials file
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'video-editor-434002-b3492f400f55.json')
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found at {credentials_path}")
            
        return service_account.Credentials.from_service_account_file(credentials_path)
    except Exception as e:
        logger.error(f"Error loading credentials: {str(e)}")
        raise

speech_client = speech.SpeechClient(credentials=get_credentials())

def extract_audio(video_path):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        video = VideoFileClip(video_path)
        if not video.audio:
            logger.warning("Video has no audio track")
            return None
            
        audio_path = tempfile.mktemp(suffix='.wav')
        
        # Convert to mono by setting audio.nchannels = 1
        video.audio.write_audiofile(audio_path, 
                                  fps=44100,  # Standard sample rate
                                  nbytes=2,   # 16-bit audio
                                  codec='pcm_s16le',  # Standard WAV codec
                                  ffmpeg_params=["-ac", "1"])  # Force mono audio
        video.close()
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(audio_path):
    try:
        with open(audio_path, 'rb') as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,  # Match the sample rate from extract_audio
            language_code="en-US",
            enable_automatic_punctuation=True,
            audio_channel_count=1,  # Explicitly set mono
        )

        response = speech_client.recognize(config=config, audio=audio)
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "
        
        return transcript.strip()
    except Exception as e:
        logger.error(f"Transcription failed for file: {audio_path}")
        logger.error(f"Error details: {str(e)}")
        return None

def generate_style_and_edits(transcript, style):
    logger.info(f"Generating style and edits for style: {style}")
    prompt = f"""
    Given the following transcript and style, generate:
    1. A stylized version of the transcript
    2. A list of video editing instructions in JSON format
    
    Transcript: {transcript}
    Style: {style}
    
    The video editing instructions should include cuts, effects, and transitions suitable for the style.
    Each instruction should have a 'type' (e.g., 'cut', 'speed', 'effect'), 'start_time', 'end_time', and 'parameters' (even if empty).
    
    Output the stylized transcript and editing instructions as a JSON object with keys 'stylized_transcript' and 'editing_instructions'.
    Do not include any markdown formatting in your response.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a video editing expert who generates stylized transcripts and editing instructions."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        logger.info(f"Raw API response: {content}")
        
        cleaned_content = clean_json_response(content)
        
        try:
            result = json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON. Error: {str(e)}. Raw content: {cleaned_content}")
            return None
        
        if not isinstance(result, dict) or 'stylized_transcript' not in result or 'editing_instructions' not in result:
            logger.error(f"Invalid response structure. Expected keys missing. Got: {result}")
            return None
        
        # Ensure all edit instructions have a 'parameters' key
        for instruction in result['editing_instructions']:
            if 'parameters' not in instruction:
                instruction['parameters'] = {}
        
        logger.info(f"Generated style and edits: {result}")
        return result
    except Exception as e:
        logger.error(f"Error generating style and edits: {str(e)}")
        return None

def apply_edits(video_path, edits):
    try:
        logger.info(f"Applying edits to {video_path}")
        video = VideoFileClip(video_path)
        final_clip = video.copy()
        
        video_duration = video.duration
        logger.info(f"Video duration: {video_duration} seconds")
        
        if not edits:
            logger.warning("No edits provided, returning original video")
            return video_path, []
            
        for edit in edits:
            try:
                edit_type = edit.get('type', 'speed')  # Default to speed if no type
                # Ensure timestamps don't exceed video duration
                start_time = min(time_to_seconds(edit.get('start_time', 0)), video_duration)
                end_time = min(time_to_seconds(edit.get('end_time', video_duration)), video_duration)
                
                # Skip if start_time >= end_time or invalid times
                if start_time >= end_time or start_time >= video_duration:
                    logger.warning(f"Skipping edit: invalid time range {start_time} to {end_time}")
                    continue
                    
                params = edit.get('parameters', {})
                
                logger.info(f"Applying {edit_type} edit from {start_time} to {end_time}")
                
                if edit_type == 'speed':
                    speed_factor = float(str(params.get('speed_factor', '1.5')).replace('x', ''))
                    before_segment = final_clip.subclip(0, start_time) if start_time > 0 else None
                    speed_segment = final_clip.subclip(start_time, end_time).speedx(speed_factor)
                    after_segment = final_clip.subclip(end_time) if end_time < video_duration else None
                    
                    # Combine segments that exist
                    segments = [seg for seg in [before_segment, speed_segment, after_segment] if seg is not None]
                    final_clip = concatenate_videoclips(segments)
                    
            except Exception as edit_error:
                logger.error(f"Error applying edit: {str(edit_error)}")
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        output_path = os.path.join(
            'output', 
            f'edited_{os.path.basename(video_path)}'
        )
        
        logger.info(f"Writing final video to {output_path}")
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            audio=True,
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        
        video.close()
        final_clip.close()
        
        return output_path, []  # Return empty list for transcripts if not needed
        
    except Exception as e:
        logger.error(f"Error applying edits: {str(e)}")
        if 'video' in locals():
            video.close()
        if 'final_clip' in locals():
            final_clip.close()
        return None, []

def process_single_file(file_path, style):
    try:
        logger.info(f"Processing single file: {file_path} with style: {style}")
        
        # Extract audio
        audio_path = extract_audio(file_path)
        if not audio_path:
            logger.error("Failed to extract audio")
            return None
            
        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        if not transcript:
            logger.error(f"Transcription failed for file: {file_path}")
            return None
            
        # Generate style and edits
        logger.info(f"Generating style and edits for style: {style}")
        result = generate_style_and_edits(transcript, style)
        if not result:
            logger.error("Failed to generate style and edits")
            return None

        # Generate document with results
        doc_path = generate_document(result)
        if not doc_path:
            logger.error("Failed to generate document")
            return None
            
        # Comment out video editing code
        '''
        # Apply edits
        logger.info(f"Applying edits to {file_path}")
        output_path, transcripts = apply_edits(file_path, result['editing_instructions'])
        if not output_path:
            logger.error(f"Failed to apply edits for file: {file_path}")
            return None
        '''
            
        return doc_path
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None
    finally:
        # Cleanup temporary files
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.error(f"Error removing audio file: {str(e)}")

def generate_document(result):
    """Generate a document with the stylized transcript and editing instructions"""
    try:
        timestamp = int(time.time())
        doc_path = os.path.join('output', f'transcript_{timestamp}.txt')
        
        with open(doc_path, 'w') as f:
            f.write("STYLIZED TRANSCRIPT\n")
            f.write("==================\n\n")
            f.write(result['stylized_transcript'])
            f.write("\n\nEDITING INSTRUCTIONS\n")
            f.write("===================\n\n")
            
            for i, instruction in enumerate(result['editing_instructions'], 1):
                f.write(f"Edit {i}:\n")
                f.write(f"Type: {instruction['type']}\n")
                f.write(f"Start Time: {instruction['start_time']}\n")
                f.write(f"End Time: {instruction['end_time']}\n")
                f.write(f"Parameters: {json.dumps(instruction['parameters'], indent=2)}\n\n")
        
        return doc_path
    except Exception as e:
        logger.error(f"Error generating document: {str(e)}")
        return None

def process_from_gcs(bucket_name, blob_name, style):
    logger.info(f"Processing file from GCS: gs://{bucket_name}/{blob_name}")
    try:
        # Download the file from GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(blob_name)[1]) as temp_file:
            blob.download_to_filename(temp_file.name)
            temp_file_path = temp_file.name
        
        # Process the file
        result = process_single_file(temp_file_path, style)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error processing file from GCS: {str(e)}")
        return None, f"Processing from GCS failed: {str(e)}"

def main(file_paths, style):
    """Process multiple video files and combine them"""
    try:
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        logger.info(f"Processing {len(file_paths)} videos with style: {style}")
        
        processed_videos = []
        all_transcripts = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing video: {file_path}")
                # Generate some basic edits if none provided
                default_edits = [{
                    'type': 'speed',
                    'start_time': 0,
                    'end_time': 5,  # First 5 seconds
                    'parameters': {'speed_factor': '1.5x'}
                }]
                
                output_path, transcripts = apply_edits(file_path, default_edits)
                if output_path:
                    processed_videos.append(output_path)
                    all_transcripts.extend(transcripts)
                    logger.info(f"Successfully processed {file_path} -> {output_path}")
                else:
                    logger.error(f"Failed to process {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(processed_videos)} videos successfully")
        
        if len(processed_videos) > 0:
            if len(processed_videos) > 1:
                logger.info("Combining multiple videos...")
                final_path = combine_videos(processed_videos)
                
                # Clean up individual processed videos
                for path in processed_videos:
                    try:
                        os.remove(path)
                        logger.info(f"Cleaned up intermediate file: {path}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {path}: {str(e)}")
                        
                return final_path, all_transcripts
            else:
                return processed_videos[0], all_transcripts
        else:
            logger.error("No videos were processed successfully")
            return None, []
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        return None, []

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_json_response(response):
    cleaned = re.sub(r'^```json\s*', '', response)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned

def time_to_seconds(time_str):
    """Convert MM:SS or integer format to seconds"""
    try:
        # If it's already a number, return it
        if isinstance(time_str, (int, float)):
            return float(time_str)
            
        # If it's a string in MM:SS format
        if ':' in str(time_str):
            minutes, seconds = map(int, str(time_str).split(':'))
            return float(minutes * 60 + seconds)
            
        # If it's a string number
        return float(time_str)
        
    except Exception as e:
        logger.error(f"Error converting time {time_str} to seconds: {str(e)}")
        return 0

def combine_videos(video_paths):
    """Combine multiple videos into one"""
    try:
        clips = [VideoFileClip(path) for path in video_paths]
        final_clip = concatenate_videoclips(clips)
        
        output_path = os.path.join('output', f'combined_video_{int(time.time())}.mp4')
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            audio=True,
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
    except Exception as e:
        logger.error(f"Error combining videos: {str(e)}")
        return None

if __name__ == "__main__":
    # Your main execution code here if needed
    pass
