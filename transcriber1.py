import os
import subprocess
import speech_recognition as sr
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account
from openai import OpenAI
from dotenv import load_dotenv
import time
import glob
import sys
import tempfile
import ast
import json
import re
from werkzeug.utils import secure_filename
import shutil
import logging
from flask import render_template
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeAudioClip, AudioFileClip, CompositeVideoClip, vfx
from moviepy.audio.fx.all import audio_fadeout, audio_fadein, volumex
import ffmpeg
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import random
from editing_presets import PRESETS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, vfx
from moviepy.video.fx.all import crop, resize
from moviepy.audio.fx.all import audio_fadeout, audio_fadein, volumex
from video_editor import VideoEditor
import traceback
import emoji

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in .env file")
client = OpenAI(api_key=openai_api_key)

# Set up Google Cloud credentials
credentials_path = "video-editor-434002-b3492f400f55.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Create a speech client with the credentials
speech_client = speech.SpeechClient(credentials=credentials)

# Utility functions
def time_to_seconds(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def check_video_file(file_path):
    command = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 \"{file_path}\""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        print(f"Error: Invalid video file: {file_path}")
        return False
    return True

# Video processing functions
def extract_audio(video_path):
    logger.info(f"Extracting audio from {video_path}")
    try:
        video = VideoFileClip(video_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1", "-ar", "16000"])
        video.close()
        logger.info(f"Audio extracted to {temp_audio_path}")
        return temp_audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio_chunk(chunk):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(chunk) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        logger.error(f"API unavailable: {str(e)}")
        return ""

def transcribe_audio(audio_path):
    logger.info(f"Transcribing audio from {audio_path}")

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    try:
        response = speech_client.recognize(config=config, audio=audio)

        transcript = []
        for result in response.results:
            alternative = result.alternatives[0]
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                transcript.append({
                    'word': word,
                    'start_time': start_time,
                    'end_time': end_time
                })

        logger.info("Transcription completed")
        return transcript
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return f"Transcription failed: {str(e)}"

def analyze_transcript(transcript):
    prompt = f"Analyze the following transcript and provide a brief summary:\n\n{transcript}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes video transcripts."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def transform_transcript(transcript, style):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Transform this transcript in a {style} style."},
                {"role": "user", "content": transcript}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error in transform_transcript: {str(e)}")

def generate_editing_instructions(transcript, style, total_duration):
    try:
        # Normalize the style name to handle slight mismatches
        normalized_style = style.lower().replace('&', 'and')
        matching_preset = next((preset for preset_name, preset in PRESETS.items() 
                                if preset_name.lower().replace('&', 'and') == normalized_style), None)
        
        if matching_preset is None:
            raise ValueError(f"No matching preset found for style: {style}")
        
        preset = matching_preset
        instructions = []
        current_time = 0

        while current_time < total_duration:
            cut_duration = random.uniform(*preset["cut_style"]["average_duration"])
            end_time = min(current_time + cut_duration, total_duration)

            cut_type = random.choices(list(preset["cut_style"]["types"].keys()),
                                      weights=list(preset["cut_style"]["types"].values()))[0]
            camera_movement = random.choices(list(preset["camera_movements"].keys()),
                                             weights=list(preset["camera_movements"].values()))[0]

            instruction = {
                "start_time": f"{int(current_time // 60):02d}:{current_time % 60:06.3f}",
                "end_time": f"{int(end_time // 60):02d}:{end_time % 60:06.3f}",
                "action": "keep",
                "cut_type": cut_type,
                "camera_movement": camera_movement
            }

            instructions.append(instruction)
            current_time = end_time

        # Add audio editing instructions
        audio_instructions = {
            "background_music": preset["audio_editing"]["music"],
            "sound_effects": preset["audio_editing"]["sound_effects"],
            "audio_ducking": preset["audio_editing"]["audio_ducking"]
        }

        return {
            "video_instructions": instructions,
            "audio_instructions": audio_instructions
        }

    except Exception as e:
        raise Exception(f"Error in generate_editing_instructions: {str(e)}")

def edit_video(input_file, style, max_duration=60):  # Limit to 60 seconds
    video = VideoFileClip(input_file)
    video = video.subclip(0, min(video.duration, max_duration))
    preset = PRESETS[style]
    
    # Apply cut style
    cut_duration = random.uniform(*preset["cut_style"]["average_duration"])
    video = video.subclip(0, min(cut_duration, video.duration))
    
    # Apply transition
    transition_funcs = list(preset["cut_style"]["transitions"].keys())
    transition_weights = list(preset["cut_style"]["transitions"].values())
    transition_func = random.choices(transition_funcs, weights=transition_weights)[0]
    video = transition_func(video)
    
    # Apply camera movement
    movement_funcs = list(preset["camera_movements"].keys())
    movement_weights = list(preset["camera_movements"].values())
    movement_func = random.choices(movement_funcs, weights=movement_weights)[0]
    video = movement_func(video)
    
    return video

# Helper function to safely apply effects
def safe_apply_effect(clip, effect_func):
    try:
        return effect_func(clip)
    except Exception as e:
        print(f"Error applying effect: {str(e)}")
        return clip  # Return original clip if effect fails

def generate_ffmpeg_instructions(editing_instructions, video_files, output_dir):
    instructions = []
    temp_files = []
    concat_list = os.path.join(output_dir, "concat_list.txt")

    if not video_files:
        raise ValueError("No video files provided")

    print(f"Number of video files: {len(video_files)}")
    print(f"Number of editing instructions: {len(editing_instructions)}")

    with open(concat_list, 'w') as f:
        for i, instruction in enumerate(editing_instructions):
            start_time = time_to_seconds(instruction['start_time'])
            duration = time_to_seconds(instruction['duration'])
            video_index = int(instruction['video_index'])

            if video_index >= len(video_files):
                print(f"Warning: video_index {video_index} is out of range. Using last available video.")
                video_index = len(video_files) - 1

            input_file = video_files[video_index]
            output_file = os.path.join(output_dir, f"temp_output_{i}.mp4")
            temp_files.append(output_file)

            command = f"ffmpeg -i \"{input_file}\" -ss {start_time:.2f} -t {duration:.2f} -c copy \"{output_file}\" -y"
            instructions.append(command)
            
            f.write(f"file '{os.path.basename(output_file)}'\n")

    final_output = os.path.join(output_dir, f"final_edited_video_{int(time.time())}.mp4")
    concat_command = f"ffmpeg -f concat -safe 0 -i \"{concat_list}\" -c copy \"{final_output}\" -y"
    instructions.append(concat_command)

    return instructions, temp_files, final_output, concat_list

def execute_ffmpeg_instructions(instructions, temp_files, final_output, concat_list):
    try:
        for i, command in enumerate(instructions):
            print(f"Executing command {i+1}/{len(instructions)}: {command}")
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"Command output: {result.stdout}")
            print(f"Command error: {result.stderr}")

            if i < len(temp_files):
                if os.path.exists(temp_files[i]) and os.path.getsize(temp_files[i]) > 0:
                    print(f"Temporary file created successfully: {temp_files[i]}")
                else:
                    print(f"Error: Temporary file not created or empty: {temp_files[i]}")
                    print(f"Contents of temp directory: {os.listdir(os.path.dirname(temp_files[i]))}")
                    raise FileNotFoundError(f"Temporary file {temp_files[i]} was not created")

        if os.path.exists(final_output):
            print(f"Final video created successfully: {final_output}")
            print(f"Final video file size: {os.path.getsize(final_output)} bytes")
        else:
            print(f"Error: Final video was not created at {final_output}")
            print(f"Contents of temp directory: {os.listdir(os.path.dirname(final_output))}")
            raise FileNotFoundError(f"Final video was not created at {final_output}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing FFmpeg command: {e}")
        print(f"Command output: {e.output}")
        print(f"Command error: {e.stderr}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(concat_list):
            os.remove(concat_list)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(video_file, creator_name, progress_callback):
    logger.debug(f"Starting process_video with file: {video_file.filename}, creator_name: {creator_name}")
    try:
        # Create a temporary directory for this video
        temp_dir = tempfile.mkdtemp()
        
        # Save the video file to the temporary directory
        temp_path = os.path.join(temp_dir, secure_filename(video_file.filename))
        video_file.save(temp_path)
        
        # Extract audio
        audio_file = extract_audio(temp_path)
        
        # Transcribe audio
        transcript = transcribe_audio(audio_file)
        
        # Update progress
        if progress_callback:
            progress_callback(1, 3, f"Processed {video_file.filename}")
        
        return temp_path, transcript, temp_dir
    except Exception as e:
        logger.exception("An error occurred in process_video")
        raise

def main(file_paths, style):
    logger.info(f"Main function called with {len(file_paths)} files and style: {style}")
    processed_clips = []
    transcripts = []
    try:
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            result = process_single_file(file_path, style)
            if result and result[0] is not None:
                styled_video, styled_transcript = result
                processed_clips.append(styled_video)
                transcripts.append(styled_transcript)
            else:
                logger.warning(f"Failed to process file: {file_path}")
        
        if processed_clips:
            # Combine all processed clips
            final_clip = concatenate_videoclips(processed_clips)
            
            # Generate a unique filename for the combined video
            output_filename = f"combined_output_{style.replace(' ', '_')}_{int(time.time())}.mp4"
            output_path = os.path.join("output", output_filename)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the final combined video
            logger.info(f"Writing final video to {output_path}")
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            logger.info(f"Combined video saved to: {output_path}")
            return output_path, transcripts
        else:
            logger.error("No videos were processed successfully")
            return None, None
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None
    finally:
        # Close all clips to free up resources
        for clip in processed_clips:
            clip.close()

def process_single_file(file_path, style):
    logger.info(f"Processing single file: {file_path} with style: {style}")
    try:
        # Extract audio and transcribe
        audio_path = extract_audio(file_path)
        if not audio_path:
            logger.error(f"Audio extraction failed for file: {file_path}")
            return None, "Audio extraction failed"
        
        transcript = transcribe_audio(audio_path)
        os.remove(audio_path)  # Clean up temporary audio file
        
        if not transcript:
            logger.error(f"Transcription failed for file: {file_path}")
            return None, "Transcription failed"
        
        # Generate style and edits
        result = generate_style_and_edits(transcript, style)
        if not result:
            logger.error(f"Failed to generate style and edits for file: {file_path}")
            return None, "Failed to generate style and edits"
        
        # Apply edits
        output_path = f"{os.path.splitext(file_path)[0]}_styled.mp4"
        edited_video_path = apply_edits(file_path, result['editing_instructions'], output_path)
        
        if not edited_video_path:
            logger.error(f"Failed to apply edits for file: {file_path}")
            return None, "Failed to apply edits"
        
        return edited_video_path, result['stylized_transcript']
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Processing failed: {str(e)}"

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_json_response(response):
    # Remove ```json and ``` from the start and end of the response
    cleaned = re.sub(r'^```json\s*', '', response)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned

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
            model="gpt-4",
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

def apply_edits(input_path, editing_instructions, output_path):
    logger.info(f"Applying edits to {input_path}")
    try:
        temp_dir = "temp_video_parts"
        os.makedirs(temp_dir, exist_ok=True)
        
        video_parts = []
        for i, instruction in enumerate(editing_instructions):
            part_output = os.path.join(temp_dir, f"part_{i}.mp4")
            
            if instruction['type'] == 'cut':
                cut_command = [
                    'ffmpeg',
                    '-i', input_path,
                    '-ss', str(instruction['start_time']),
                    '-to', str(instruction['end_time']),
                    '-c', 'copy',
                    part_output
                ]
                subprocess.run(cut_command, check=True, capture_output=True)
                video_parts.append(part_output)
            
            elif instruction['type'] == 'speed':
                speed_factor = instruction['parameters'].get('playback_speed', 1.0)
                speed_command = [
                    'ffmpeg',
                    '-i', input_path,
                    '-ss', str(instruction['start_time']),
                    '-to', str(instruction['end_time']),
                    '-filter:v', f"setpts={1/speed_factor}*PTS",
                    '-filter:a', f"atempo={speed_factor}",
                    part_output
                ]
                subprocess.run(speed_command, check=True, capture_output=True)
                video_parts.append(part_output)
            
            elif instruction['type'] == 'effect':
                effect_type = instruction['parameters'].get('type', '')
                if effect_type == 'blur':
                    blur_amount = instruction['parameters'].get('amount', 5)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"boxblur={blur_amount}:1",
                        part_output
                    ]
                elif effect_type == 'brightness':
                    brightness = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"brightness={brightness}",
                        part_output
                    ]
                elif effect_type == 'contrast':
                    contrast = instruction['parameters'].get('amount', 1)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"contrast={contrast}",
                        part_output
                    ]
                elif effect_type == 'saturation':
                    saturation = instruction['parameters'].get('amount', 1)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"saturation={saturation}",
                        part_output
                    ]
                elif effect_type == 'hue':
                    hue = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"hue={hue}",
                        part_output
                    ]
                elif effect_type == 'noise':
                    noise_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"noise={noise_amount}",
                        part_output
                    ]
                elif effect_type == 'sharpen':
                    sharpen_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"unsharp={sharpen_amount}",
                        part_output
                    ]
                elif effect_type == 'edge':
                    edge_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"edgedetect={edge_amount}",
                        part_output
                    ]
                elif effect_type == 'color':
                    color_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"color={color_amount}",
                        part_output
                    ]
                elif effect_type == 'gray':
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', 'gray',
                        part_output
                    ]
                elif effect_type == 'invert':
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', 'negate',
                        part_output
                    ]
                elif effect_type == 'sepia':
                    sepia_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"sepia={sepia_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
                        '-filter:v', f"threshold={threshold_amount}",
                        part_output
                    ]
                elif effect_type == 'threshold':
                    threshold_amount = instruction['parameters'].get('amount', 0)
                    effect_command = [
                        'ffmpeg',
                        '-i', input_path,
                        '-ss', str(instruction['start_time']),
                        '-to', str(instruction['end_time']),
            '-filter_complex', filter_complex_str,
            '-map', '[outv]',
            '-c:a', 'copy',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        logger.info(f"Edited video saved to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        logger.error(f"Error applying edits: {str(e)}")
        return None

def style_transcript_fast(transcript):
    # Add quick cuts and energetic transitions
    words = transcript.split()
    styled = []
    for i, word in enumerate(words):
        if i % 5 == 0:
            styled.append(f"[CUT] {word.upper()}")
        else:
            styled.append(word)
    return " ".join(styled)

def style_transcript_moderate(transcript):
    # Add moderate pacing and transitions
    sentences = transcript.split('.')
    styled = []
    for i, sentence in enumerate(sentences):
        if i % 2 == 0:
            styled.append(f"{sentence.strip()}. [FADE]")
        else:
            styled.append(sentence.strip())
    return " ".join(styled)

def style_transcript_slow(transcript):
    # Add smooth transitions and emphasis
    words = transcript.split()
    styled = []
    for i, word in enumerate(words):
        if i % 10 == 0:
            styled.append(f"[SLOW ZOOM] {word}")
        else:
            styled.append(word)
    return " ".join(styled)

# Add a new function to apply audio editing
def apply_audio_editing(clip, audio_instructions, style):
    audio = clip.audio
    voiceover = audio_instructions["voiceover"]
    
    final_audio = PRESETS[style]["audio_editing"]["audio_ducking"](voiceover, clip.duration)
    return clip.set_audio(final_audio)

# Add placeholder functions for cut types and camera movements
def apply_cut_type(clip, cut_type):
    if cut_type == "Fast Cut":
        return clip
    elif cut_type == "Swipe Cut":
        return get_available_effect("slide_out", fallback=vfx.fadeout)(clip, duration=0.5, side="left")
    elif cut_type == "Fade In/Out":
        return get_available_effect("fadeout", fallback=vfx.fadeout)(clip, duration=0.5).fx(get_available_effect("fadein", fallback=vfx.fadein), duration=0.5)
    elif cut_type == "Dissolve":
        return get_available_effect("crossfadein", fallback=vfx.fadein)(clip, duration=0.5)
    else:
        return clip

def apply_camera_movement(clip, camera_movement):
    if camera_movement == "Handheld":
        return get_available_effect("shake", fallback=lambda c: c)(clip, amplitude=5)
    elif camera_movement == "Smooth pan or tilt":
        return get_available_effect("speedx", fallback=lambda c: c)(clip, factor=1.1)
    elif camera_movement == "Static with zoom":
        return get_available_effect("zoom", fallback=lambda c: c)(clip, zoom_ratio=1.1)
    else:
        return clip

# Custom shake effect
def shake(clip, amplitude=1, duration=None):
    def transform(get_frame, t):
        frame = get_frame(t)
        dx = int(amplitude * np.sin(t * np.pi * 2))
        dy = int(amplitude * np.cos(t * np.pi * 2))
        return frame.copy().rotate(dx)  # Rotate instead of crop for a shake effect
    return clip.fl(transform)

# Add the custom shake effect to vfx
vfx.shake = shake

def get_available_effect(effect_name, fallback=None):
    if hasattr(vfx, effect_name):
        return getattr(vfx, effect_name)
    elif effect_name in globals():
        return globals()[effect_name]
    elif fallback:
        print(f"Warning: Effect '{effect_name}' not found. Using fallback.")
        return fallback
    else:
        print(f"Warning: Effect '{effect_name}' not found and no fallback provided.")
        return lambda clip: clip  # Return a no-op function

# Custom effects
def slide_out(clip, duration=1, side="right"):
    w, h = clip.size
    if side in ["right", "left"]:
        move = lambda t: ('x', int(t * w))
    else:
        move = lambda t: ('y', int(t * h))
    return clip.fx(vfx.slide_out, duration, move)

# Add custom effects to vfx
vfx.slide_out = slide_out

def apply_transition(clip, transition_func):
    return transition_func(clip)

def apply_camera_movement(clip, movement_func):
    return movement_func(clip)

def transcribe(file):
    # Your transcription logic here
    pass

# Make sure to export both main and transcribe
__all__ = ['main', 'transcribe']
