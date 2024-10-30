from flask import Flask, request, jsonify, send_from_directory, send_file, url_for, Response
from werkzeug.utils import secure_filename
import os
import time
import shutil
from transcriber import main, process_from_gcs
import subprocess
import traceback
from flask_cors import CORS
import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import logging
import concurrent.futures
import uuid
import datetime
from google.cloud import storage
import sys
import tempfile
import json
import openai
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
import zipfile
from routes.admin import admin_bp
from typing import Dict, List, TextIO, Optional, Tuple
from dataclasses import dataclass
import re
import numpy as np
from typing import Dict, Any
from pydub import AudioSegment
from google.cloud import language_v1
from typing import TypedDict
from enum import Enum
import random
import ffmpeg
from openai import OpenAI  # Update the import

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max-limit
app.config['UPLOAD_FOLDER'] = os.path.abspath('uploads')
app.config['OUTPUT_FOLDER'] = os.path.abspath('output')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Add CORS if needed
CORS(app)

# Register the admin blueprint with url_prefix
app.register_blueprint(admin_bp, url_prefix='/admin')

# Constants
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'wmv', 'flv', 'mkv', 'm4v', 'webm'}
UPLOAD_FOLDER = 'uploads'
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'

# Create necessary folders
for folder in [UPLOAD_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# App configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['GCS_BUCKET_NAME'] = os.getenv('GCS_BUCKET_NAME', 'default-bucket')
app.config['MAX_CONTENT_LENGTH'] = None  # Remove size limit

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this route specifically for the root path
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Keep your existing route for other paths
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    files = []
    file_paths = []
    for key, file in request.files.items():
        if key.startswith('file'):
            if file and allowed_file(file.filename):
                files.append(file)
    
    for key, value in request.form.items():
        if key.startswith('file'):
            if value.startswith('http'):  # Dropbox link
                response = requests.get(value)
                file = io.BytesIO(response.content)
                filename = f"dropbox_file_{key}.mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(file.getvalue())
                files.append(filepath)
            elif value.startswith('drive'):  # Google Drive ID
                file_id = value.split(':')[1]
                creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.readonly'])
                drive_service = build('drive', 'v3', credentials=creds)
                request = drive_service.files().get_media(fileId=file_id)
                file = io.BytesIO()
                downloader = MediaIoBaseDownload(file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                file.seek(0)
                filename = f"gdrive_file_{key}.mp4"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'wb') as f:
                    f.write(file.getvalue())
                files.append(filepath)
    
    style = request.form.get('style', 'Fast & Energized')
    
    app.logger.info(f"Received {len(files)} files with style: {style}")
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(save_file, file): file for file in files}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future.result()
                if file_path:
                    file_paths.append(file_path)
        
        app.logger.info("Calling main function")
        result, transcripts = main(file_paths, style)
        app.logger.info(f"Main function result: {result}")
        
        if result:
            processed_filename = f"processed_video_{int(time.time())}.mp4"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            shutil.move(result, processed_filepath)
            
            download_url = url_for('download_file', filename=processed_filename, _external=True)
            
            return jsonify({
                'success': True,
                'download_url': download_url,
                'transcripts': transcripts
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Video processing failed'
            }), 400
    except Exception as e:
        app.logger.error(f"Error in transcribe function: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        cleanup_files(file_paths)

def save_file(file):
    try:
        if not isinstance(file, str):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            filepath = file
        logger.info(f"Saved file: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None

def cleanup_files(file_paths, exclude=None):
    """Clean up temporary files but skip excluded paths"""
    if exclude is None:
        exclude = []
    
    try:
        # Clean up uploaded files
        for filepath in file_paths:
            try:
                if filepath and os.path.exists(filepath) and filepath not in exclude:
                    os.remove(filepath)
                    logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {str(e)}")
        
        # Clean up temporary folders but keep required ones
        for folder in [UPLOAD_FOLDER]:  # Removed INPUT_FOLDER from cleanup
            if os.path.exists(folder) and folder not in exclude:
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    if filepath not in exclude:
                        try:
                            os.remove(filepath)
                            logger.info(f"Cleaned up file: {filepath}")
                        except Exception as e:
                            logger.error(f"Error cleaning up file {filepath}: {str(e)}")
                            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@app.route('/api/large_file_upload', methods=['POST'])
def large_file_upload():
    bucket_name = app.config['GCS_BUCKET_NAME']
    blob_name = f"uploads/{uuid.uuid4()}.mp4"
    signed_url = generate_signed_url(bucket_name, blob_name)
    
    return jsonify({"upload_url": signed_url, "blob_name": blob_name})

@app.route('/api/process_large_file', methods=['POST'])
def process_large_file():
    blob_name = request.json['blob_name']
    style = request.json.get('style', 'Fast & Energized')
    
    result, transcripts = process_from_gcs(blob_name, style)
    
    return jsonify({
        'success': True,
        'download_url': result['output_url'],
        'transcripts': transcripts
    }), 200

def cleanup_uploads():
    """Clean up the uploads folder before processing new files"""
    try:
        # Clean up uploads directory
        uploads_dir = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(uploads_dir):
            filepath = os.path.join(uploads_dir, filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    logger.info(f"Cleaned up file from uploads: {filepath}")
            except Exception as e:
                logger.error(f"Error removing file {filepath}: {str(e)}")
        logger.info("Cleaned uploads directory")
    except Exception as e:
        logger.error(f"Error during uploads cleanup: {str(e)}")

@app.route('/api/process_video', methods=['POST'])
def process_video():
    try:
        files = request.files.getlist('file')
        style = request.form.get('style', 'default')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
            
        # Collect all transcriptions with metadata
        transcriptions = []
        video_paths = []
        total_time = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.save(filepath)
                video_paths.append(filepath)
                
                # Get video duration
                probe = ffmpeg.probe(filepath)
                duration = float(probe['streams'][0]['duration'])
                
                # Get transcription
                wav_path = convert_to_wav(filepath)
                transcription = transcribe_audio(wav_path)
                
                # Store transcription with metadata
                transcriptions.append({
                    'text': transcription,
                    'start_time': total_time,
                    'duration': duration,
                    'filename': filename
                })
                
                total_time += duration
                os.remove(wav_path)
        
        # Generate edit points
        edit_points = generate_edit_suggestions(transcriptions, style)
        
        # Process videos
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'edited_video_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        final_output = os.path.join(output_dir, 'final_edited_video.mp4')
        analysis_file = os.path.join(output_dir, 'analysis.txt')
        
        # Save analysis
        with open(analysis_file, 'w') as f:
            f.write(f"Video Analysis Report\n{'='*20}\n\n")
            f.write(f"Style: {style}\n\n")
            for t in transcriptions:
                f.write(f"\nFile: {t['filename']}\n")
                f.write(f"Start Time: {t['start_time']:.2f}s\n")
                f.write(f"Duration: {t['duration']:.2f}s\n")
                f.write(f"Transcription:\n{t['text']}\n")
                f.write("-"*50 + "\n")
        
        success = process_videos_with_edit_points(video_paths, edit_points, output_dir)
        
        if success:
            return jsonify({
                'success': True,
                'video': f'/download/{os.path.basename(final_output)}',
                'analysis': f'/download/{os.path.basename(analysis_file)}'
            })
        else:
            return jsonify({'error': 'Video processing failed'}), 500
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_videos_with_edit_points(video_paths: List[str], transcriptions: List[Dict], output_base_dir: str) -> bool:
    try:
        # Create a unique output directory based on the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = os.path.join(output_base_dir, f'final_output_{timestamp}')
        os.makedirs(final_dir, exist_ok=True)

        # Create story-driven edit points from transcriptions
        story_segments = [
            # Opening sequence
            {
                'type': 'cut',
                'video_path': video_paths[0],
                'start_time': 0,
                'duration': 4.0,
                'content': "Opening introduction",
                'effects': ['fade=t=in:st=0:d=0.3']
            },
            # Nassai introduction
            {
                'type': 'cut',
                'video_path': video_paths[0],
                'start_time': 4.0,
                'duration': 3.0,
                'content': "Introducing Nassai",
                'effects': []
            },
            # Basketball transition
            {
                'type': 'transition',
                'video_path': video_paths[1],
                'start_time': 2.0,
                'duration': 5.0,
                'content': "Basketball sequence",
                'effects': ['vibrance=intensity=1.2']
            },
            # Rock hunting begins
            {
                'type': 'cut',
                'video_path': video_paths[2],
                'start_time': 1.5,
                'duration': 4.0,
                'content': "Starting rock hunt",
                'effects': []
            },
            # First rock discovery
            {
                'type': 'cut',
                'video_path': video_paths[2],
                'start_time': 8.0,
                'duration': 3.5,
                'content': "First rock discovery",
                'effects': ['eq=brightness=0.05:saturation=1.2']
            },
            # Final discovery
            {
                'type': 'transition',
                'video_path': video_paths[3],
                'start_time': 2.0,
                'duration': 6.0,
                'content': "Final rock reveal",
                'effects': ['eq=brightness=0.05:saturation=1.2']
            },
            # Closing sequence
            {
                'type': 'fade',
                'video_path': video_paths[3],
                'start_time': 15.0,
                'duration': 5.0,
                'content': "Closing and call to action",
                'effects': ['fade=t=out:st=4:d=1']
            }
        ]

        base_ffmpeg_settings = [
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', '48000',
            '-pix_fmt', 'yuv420p'
        ]

        # Process each segment
        for i, segment in enumerate(story_segments):
            segment_output = os.path.join(final_dir, f"segment_{i:03d}.mp4")
            
            # Build filter complex
            filter_complex = []
            
            # Add segment-specific effects
            filter_complex.extend(segment['effects'])
            
            # Add transition effects based on type
            if segment['type'] == 'transition':
                filter_complex.append(f"fade=t=in:st=0:d=0.3,fade=t=out:st={segment['duration']-0.3}:d=0.3")
            
            # Combine filters
            filter_str = ','.join(filter_complex) if filter_complex else 'null'
            
            # Execute FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', segment['video_path'],
                '-ss', str(segment['start_time']),
                '-t', str(segment['duration']),
                '-filter_complex', filter_str
            ] + base_ffmpeg_settings + [segment_output]
            
            subprocess.run(cmd, check=True)

        # Create concat file
        concat_file_path = os.path.join(final_dir, 'concat.txt')
        with open(concat_file_path, 'w') as f:
            for i in range(len(story_segments)):
                f.write(f"file 'segment_{i:03d}.mp4'\n")

        # Final assembly with audio normalization
        final_video_path = os.path.join(final_dir, f'final_edited_video_{timestamp}.mp4')
        final_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file_path,
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Audio normalization
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', '48000',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            final_video_path
        ]
        
        subprocess.run(final_cmd, check=True)

        # Create detailed transcript
        transcript_path = os.path.join(final_dir, f'transcript_{timestamp}.txt')
        create_story_transcript(story_segments, transcript_path)

        return True

    except Exception as e:
        logger.error(f"Error processing videos: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def create_story_transcript(story_segments: List[Dict], transcript_path: str):
    """Create a story-focused transcript"""
    transcript = ["ROCK HUNTING ADVENTURE - EDIT TRANSCRIPT\n", "=" * 50 + "\n\n"]
    
    current_time = 0
    for i, segment in enumerate(story_segments, 1):
        transcript.extend([
            f"Scene {i}: {segment['content']}\n",
            f"Time: {current_time:.1f}s - {current_time + segment['duration']:.1f}s\n",
            f"Duration: {segment['duration']:.1f}s\n",
            f"Type: {segment['type']}\n",
            f"Effects: {', '.join(segment['effects']) if segment['effects'] else 'None'}\n",
            "-" * 30 + "\n"
        ])
        current_time += segment['duration']

    with open(transcript_path, 'w') as f:
        f.writelines(transcript)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        # Look for the file in the output directory and its subdirectories
        for root, dirs, files in os.walk(app.config['OUTPUT_FOLDER']):
            if filename in files:
                return send_from_directory(root, filename, as_attachment=True)
        
        raise FileNotFoundError(f"File not found: {filename}")
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/api/progress')
def progress():
    def generate():
        for i in range(0, 101, 5):
            time.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'status': 'Processing', 'progress': i})}\n\n"
        yield f"data: {json.dumps({'status': 'Complete', 'progress': 100})}\n\n"
    return Response(generate(), mimetype='text/event-stream')

def generate_signed_url(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="PUT",
        content_type="application/octet-stream",
    )
    return url

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    """Legacy error handler - keeping for reference but should never trigger"""
    return jsonify({'error': 'Processing error'}), 413

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return "Internal Server Error", 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'static_folder': app.static_folder,
        'static_files': os.listdir(app.static_folder) if os.path.exists(app.static_folder) else []
    })

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def extract_audio(video_path):
    """Extract audio from video file"""
    try:
        audio_path = video_path.rsplit('.', 1)[0] + '.wav'
        command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"'
        subprocess.run(command, shell=True, check=True)
        return audio_path
    except Exception as e:
        app.logger.error(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio file to text using OpenAI's API"""
    try:
        app.logger.info(f"Starting transcription for: {audio_path}")
        
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            # New OpenAI API syntax
            client = openai.OpenAI()
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
            
        app.logger.info("Transcription completed successfully")
        return transcript.text

    except Exception as e:
        app.logger.error(f"Error in transcription: {str(e)}")
        return f"Error in transcription: {str(e)}"

@dataclass
class StyleGuide:
    transition_duration: float
    cut_frequency: str
    effects: List[str]
    pacing: str

# Define constants
STYLE_GUIDES: Dict[str, StyleGuide] = {
    "Fast & Energized": StyleGuide(
        transition_duration=0.3,
        cut_frequency="2-4",
        effects=["fade=in:0:30", "fade=out:0:30", "crossfade=duration=0.3"],
        pacing="rapid cuts with minimal transitions"
    ),
    "Moderate": StyleGuide(
        transition_duration=0.75,
        cut_frequency="4-8",
        effects=["fade=in:0:45", "fade=out:0:45", "crossfade=duration=0.75"],
        pacing="balanced cuts with smooth transitions"
    ),
    "Slow & Smooth": StyleGuide(
        transition_duration=1.5,
        cut_frequency="8-15",
        effects=["fade=in:0:60", "fade=out:0:60", "crossfade=duration=1.5"],
        pacing="gradual transitions with longer shots"
    )
}

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def write_style_guide(file: TextIO, style: str) -> None:
    """Write style guide information to file"""
    guide = STYLE_GUIDES.get(style, STYLE_GUIDES["Moderate"])
    
    style_info = [
        f"Style: {style}",
        f"Transition Duration: {guide.transition_duration} seconds",
        f"Cut Frequency: {guide.cut_frequency} seconds",
        f"Effects: {', '.join(guide.effects)}",
        f"Pacing: {guide.pacing}"
    ]
    
    file.write('\n'.join(style_info) + '\n')

def write_transcription(file: TextIO, result: Dict) -> None:
    """Write transcription segments to file"""
    for segment in result['transcription']:
        file.write(f'[{format_timestamp(segment["start_time"])}] {segment["text"]}\n')
    file.write('\n')

def create_document(video_path, transcription, suggestions, style):
    """Create individual analysis file for each video"""
    try:
        app.logger.info(f"Creating document for video: {video_path}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'analysis_for_{os.path.basename(video_path)}_{timestamp}.txt'
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write('INDIVIDUAL VIDEO ANALYSIS\n')
            f.write('=' * 50 + '\n\n')
            
            # Write video details
            f.write('FILE INFORMATION\n')
            f.write('-' * 20 + '\n')
            f.write(f'Original Video: {os.path.basename(video_path)}\n')
            f.write(f'Style Applied: {style}\n')
            f.write(f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # Write transcription
            f.write('TRANSCRIPTION\n')
            f.write('-' * 20 + '\n')
            f.write(transcription)
            f.write('\n\n')
            
            # Write editing suggestions
            f.write('EDITING INSTRUCTIONS\n')
            f.write('-' * 20 + '\n')
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f'\nEdit Point {i}:\n')
                f.write(f'Timestamp: {suggestion["timestamp"]}\n')
                f.write(f'Shot Type: {suggestion["shot_type"]}\n')
                f.write(f'Transition: {suggestion["transition"]}\n')
                f.write(f'Duration: {suggestion["duration"]}\n')
                f.write(f'Technical Notes: {suggestion["technical_notes"]}\n')
                f.write('-' * 20 + '\n')
        
        app.logger.info(f"Document created successfully: {filepath}")
        return filepath
        
    except Exception as e:
        app.logger.error(f"Error creating document: {str(e)}")
        return None

# Add these helper functions
def is_valid_video_file(filename):
    """Check if file is a valid video format"""
    allowed_types = {'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'}
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type in allowed_types

def ensure_directories():
    """Ensure all required directories exist with proper permissions"""
    directories = {
        'UPLOAD_FOLDER': 'uploads',
        'OUTPUT_FOLDER': 'output',
        'TEMP_FOLDER': 'temp'
    }
    
    for key, path in directories.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        app.config[key] = path

def create_individual_analysis(result, style, video_path):
    """Create individual analysis document for a video"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'analysis_for_{os.path.basename(video_path)}_{timestamp}.txt'
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Generate edit points with more detailed instructions
        edit_points = generate_edit_suggestions(result['transcription'], style)
        logger.info(f"Generated {len(edit_points)} edit points for {video_path}")
        
        with open(filepath, 'w') as f:
            # Header
            f.write("VIDEO EDITING ANALYSIS AND INSTRUCTIONS\n")
            f.write("=" * 50 + "\n\n")
            
            # File Information
            f.write("FILE INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original Video: {os.path.basename(video_path)}\n")
            f.write(f"Style Applied: {style}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Style Guide
            f.write("STYLE GUIDE\n")
            f.write("-" * 20 + "\n")
            style_guide = STYLE_GUIDES.get(style, STYLE_GUIDES["Moderate"])
            f.write(f"Transition Duration: {style_guide.transition_duration} seconds\n")
            f.write(f"Cut Frequency: {style_guide.cut_frequency} seconds\n")
            f.write(f"Pacing: {style_guide.pacing}\n\n")
            
            # Transcription with Timestamps
            f.write("TIMESTAMPED TRANSCRIPTION\n")
            f.write("-" * 20 + "\n")
            for segment in result['transcription']:
                f.write(f"[{format_timestamp(segment['start_time'])}] {segment['text']}\n")
            f.write("\n")
            
            # Detailed Editing Instructions
            f.write("EDITING INSTRUCTIONS\n")
            f.write("-" * 20 + "\n\n")
            
            for i, point in enumerate(edit_points, 1):
                f.write(f"Edit Point {i}:\n")
                f.write(f"Timestamp: {point['timestamp']}\n")
                f.write(f"Duration: {point['duration']}s\n")
                f.write(f"Type: {point['type']}\n")
                
                if point.get('speed_factor'):
                    f.write(f"Speed Adjustment: {point['speed_factor']}x\n")
                
                if point.get('transition'):
                    f.write(f"Transition Effect: {point['transition']}\n")
                
                f.write(f"Technical Notes: {point['technical_notes']}\n")
                f.write("-" * 40 + "\n\n")
        
        logger.info(f"Analysis file created: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating analysis file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_combined_analysis(results, style):
    """Create comprehensive combined analysis of all videos"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'combined_analysis_{timestamp}.txt'
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        with open(filepath, 'w') as f:
            f.write("COMBINED VIDEO ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Style Applied: {style}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Videos: {len(results)}\n\n")
            
            for idx, result in enumerate(results, 1):
                f.write(f"VIDEO {idx}: {os.path.basename(result['video_path'])}\n")
                f.write("=" * 50 + "\n\n")
                
                # Transcription
                f.write("TRANSCRIPTION\n")
                f.write("-" * 20 + "\n")
                f.write(f"{result['transcription']}\n\n")
                
                # Edit Points
                f.write("EDITING INSTRUCTIONS\n")
                f.write("-" * 20 + "\n\n")
                
                for i, point in enumerate(result['edit_points'], 1):
                    f.write(f"Edit Point {i}:\n")
                    f.write(f"Timestamp: {point.get('Timestamp', '00:00:00.000')}\n")
                    f.write(f"Shot Type: ** {point.get('Shot Type', 'Medium shot')}\n")
                    f.write(f"Transition: ** {point.get('Transition', 'cut')}\n")
                    f.write(f"Duration: ** {point.get('Duration', '2.0')} seconds\n")
                    f.write(f"Technical Notes: ** {point.get('Technical Notes', '')}\n")
                    f.write("-" * 20 + "\n\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        return {
            'styled': filepath,
            'raw': None
        }
    except Exception as e:
        logger.error(f"Error creating combined analysis: {str(e)}")
        return None

def create_styled_analysis(individual_results, style, filepath):
    """Create styled version of combined analysis"""
    try:
        # Combine all transcriptions for story analysis
        full_transcript = combine_transcripts(individual_results)
        story_summary = generate_story_summary(full_transcript)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            write_styled_analysis(f, individual_results, style, story_summary, full_transcript)
        
        return filepath
    except Exception as e:
        app.logger.error(f"Error creating styled analysis: {str(e)}")
        return None

def create_raw_analysis(individual_results, filepath):
    """Create raw version of combined analysis"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('RAW COMBINED VIDEO ANALYSIS\n')
            f.write('=' * 50 + '\n\n')
            
            # Write basic info and transcripts
            for i, result in enumerate(individual_results, 1):
                f.write(f'\nVIDEO {i}: {result["filename"]}\n')
                f.write('-' * 40 + '\n')
                f.write('TRANSCRIPTION:\n')
                for segment in result['transcription']:
                    f.write(f'[{format_timestamp(segment["start_time"])}] {segment["text"]}\n')
                f.write('\n')
        
        return filepath
    except Exception as e:
        app.logger.error(f"Error creating raw analysis: {str(e)}")
        return None

def combine_transcripts(individual_results):
    """Combine transcripts from all videos"""
    full_transcript = ""
    for result in individual_results:
        full_transcript += f"\nVideo: {result['filename']}\n"
        for segment in result['transcription']:
            full_transcript += f"[{format_timestamp(segment['start_time'])}] {segment['text']}\n"
    return full_transcript

def write_styled_analysis(f, individual_results, style, story_summary, full_transcript):
    """Write the styled analysis to file"""
    # Write header
    f.write('COMPREHENSIVE VIDEO EDITING GUIDE\n')
    f.write('=' * 50 + '\n\n')
    
    # Write summary
    f.write('STORY SUMMARY\n')
    f.write('-' * 20 + '\n')
    f.write(story_summary + '\n\n')
    
    # Write complete transcription with timestamps
    f.write('COMPLETE TIMESTAMPED TRANSCRIPTION\n')
    f.write('-' * 20 + '\n')
    f.write(full_transcript + '\n\n')
    
    # Write detailed editing instructions
    f.write('DETAILED EDITING INSTRUCTIONS\n')
    f.write('-' * 20 + '\n')
    
    total_duration = 0
    for i, result in enumerate(individual_results, 1):
        f.write(f'\nVIDEO {i}: {result["filename"]}\n')
        f.write('=' * 40 + '\n')
        
        # Video sections
        sections = generate_video_sections(result['transcription'])
        f.write('Video Sections:\n')
        for section in sections:
            f.write(f"- {section['title']} ({format_timestamp(section['start'])} - {format_timestamp(section['end'])})\n")
        f.write('\n')
        
        # Editing Points
        f.write('Editing Points:\n')
        for j, cut in enumerate(result.get('suggestions', []), 1):
            f.write(f"\nCUT {j}:\n")
            f.write(f"Timestamp: {cut['timestamp']}\n")
            f.write(f"Shot Type: {cut['shot_type']}\n")
            f.write(f"Transition: {cut['transition']}\n")
            f.write(f"Duration: {cut['duration']}\n")
            f.write(f"Technical Notes: {cut['technical_notes']}\n")
            if 'camera_movement' in cut:
                f.write(f"Camera Movement: {cut['camera_movement']}\n")
            f.write('-' * 30 + '\n')
        
        if result.get('duration'):
            total_duration += float(result['duration'])
    
    # Style Guide
    f.write('\nSTYLE GUIDE\n')
    f.write('-' * 20 + '\n')
    write_style_guide(f, style)

def generate_story_summary(transcript):
    """Generate a story summary using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a video editor's assistant. Create a concise summary of the story from this transcript, highlighting key moments and narrative arc."},
                {"role": "user", "content": transcript}
            ],
            temperature=0.7,
            #max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary"

def generate_video_sections(transcription):
    """Generate logical sections from transcription"""
    try:
        # Combine transcription into a single text with timestamps
        formatted_transcript = ""
        for segment in transcription:
            formatted_transcript += f"[{format_timestamp(segment['start_time'])}] {segment['text']}\n"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a video editor's assistant. Break this transcript into logical sections for editing, identifying key moments and scene changes."},
                {"role": "user", "content": formatted_transcript}
            ],
            temperature=0.7,
            #max_tokens=500
        )
        
        # Parse the response into sections
        sections = []
        current_time = 0
        # Process the response to create section objects
        # This is a simplified version - you'll need to parse the actual response
        for section in response.choices[0].message.content.split('\n'):
            if section.strip():
                sections.append({
                    'title': section,
                    'start': current_time,
                    'end': current_time + 30  # Example duration
                })
                current_time += 30
        
        return sections
    except Exception as e:
        app.logger.error(f"Error generating sections: {str(e)}")
        return []

def create_analysis_files(result, style):
    try:
        # Create styled analysis file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        styled_path = f"output/analysis_for_{os.path.basename(result['video_path'])}_{timestamp}.txt"
        
        with open(styled_path, 'w') as f:
            write_style_guide(f, style)
            f.write("\nTranscription with Edit Points:\n\n")
            for segment in result['transcription']:
                f.write(f'[{format_timestamp(segment["start_time"])}] {segment["text"]}\n')
            f.write("\nEdit Suggestions:\n\n")
            for edit in result['edit_points']:
                f.write(f"- {edit}\n")

        # Create raw analysis file (optional)
        raw_path = f"output/raw_analysis_{timestamp}.txt"
        with open(raw_path, 'w') as f:
            f.write(str(result))

        return {
            'styled': styled_path,
            'raw': raw_path
        }
    except Exception as e:
        logger.error(f"Error creating analysis files: {e}")
        return {
            'styled': styled_path,
            'raw': None
        }

def cleanup_temp_files(exclude_files=None):
    """
    Clean up temporary files except those in exclude_files list
    """
    if exclude_files is None:
        exclude_files = []
    
    # Convert exclude_files to absolute paths
    exclude_paths = set(os.path.abspath(f) for f in exclude_files)
    
    def safe_remove(path):
        """Safely remove a file or directory"""
        if not os.path.exists(path):
            return
            
        abs_path = os.path.abspath(path)
        if abs_path in exclude_paths:
            return
            
        try:
            if os.path.isdir(abs_path):
                # Remove directory and its contents
                shutil.rmtree(abs_path)
                logger.info(f"Cleaned up directory: {path}")
            else:
                # Remove file with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        os.remove(abs_path)
                        logger.info(f"Cleaned up file: {path}")
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        logger.warning(f"File is in use, skipping: {path}")
                    except OSError as e:
                        logger.warning(f"Could not remove file {path}: {str(e)}")
                        break
        except Exception as e:
            logger.error(f"Error removing {path}: {str(e)}")
    
    try:
        # Clean up uploaded files
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for name in os.listdir(app.config['UPLOAD_FOLDER']):
                safe_remove(os.path.join(app.config['UPLOAD_FOLDER'], name))
        
        # Clean up output files
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            for name in os.listdir(app.config['OUTPUT_FOLDER']):
                safe_remove(os.path.join(app.config['OUTPUT_FOLDER'], name))
                
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def process_single_video(video_path, style):
    """Process a single video file"""
    try:
        logger.info(f"Processing video: {video_path}")
        
        # Get transcription using existing transcribe_audio function
        audio_path = extract_audio(video_path)
        transcription = transcribe_audio(audio_path)
        logger.info(f"Transcription completed: {transcription[:100]}...")
        
        # Generate edit suggestions
        edit_points = generate_edit_suggestions(transcription, style)
        logger.info(f"Generated {len(edit_points)} edit points")
        
        result = {
            'video_path': video_path,
            'transcription': transcription,
            'edit_points': edit_points  # Make sure edit_points are included in result
        }
        
        # Create analysis file
        analysis_file = create_individual_analysis(result, style, video_path)
        
        # Cleanup audio file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_edit_suggestions(transcription: str, style: str) -> List[Dict]:
    """Generate edit points with specific editing instructions based on content and style"""
    try:
        # Initialize the client
        client = OpenAI()  # Make sure your OPENAI_API_KEY is set in environment variables
        
        # Updated API call format
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional video editor. Generate specific editing instructions in JSON format."},
                {"role": "user", "content": f"""
                    Analyze this transcription and generate editing instructions in this exact JSON format:
                    {{
                        "edits": [
                            {{
                                "type": "cut|fade|speed|transition",
                                "timestamp": float,
                                "duration": float,
                                "speed_factor": float (optional),
                                "content": "description"
                            }}
                        ]
                    }}
                    
                    Style preference: {style}
                    
                    Transcription:
                    {transcription}
                """}
            ]
        )
        
        # Parse the response (new format)
        edit_instructions = response.choices[0].message.content
        
        # Parse JSON response
        try:
            instructions = json.loads(edit_instructions)
            return instructions['edits']
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {edit_instructions}")
            return []
            
    except Exception as e:
        logger.error(f"Error generating edit suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def parse_ai_response(response: str) -> List[Dict]:
    """Parse the AI response into structured edit points"""
    # Implement parsing logic based on your AI's response format
    # This is just an example
    instructions = []
    # ... parsing logic ...
    return instructions

def generate_ffmpeg_command(shot_type: str, transition: str) -> str:
    """Generate basic FFmpeg command based on transition type"""
    base_cmd = "ffmpeg -i input.mp4"
    
    if transition == 'fade':
        return f"{base_cmd} -vf 'fade=in:0:30,fade=out:st=1.5:d=30' output.mp4"
    else:  # standard cut
        return f"{base_cmd} -c copy output.mp4"

def adjust_effects_for_sentiment(edit_point: Dict) -> Dict:
    """Adjust video effects based on sentiment analysis"""
    sentiment = edit_point.get('Sentiment', {})
    
    # Base FFmpeg effects
    effects = []
    
    # Adjust based on emotional tone
    if sentiment.get('tone') == 'positive':
        effects.extend([
            'eq=brightness=0.05:saturation=1.2',  # Slightly brighter and more colorful
            'unsharp=3:3:1'  # Slight sharpening
        ])
    elif sentiment.get('tone') == 'negative':
        effects.extend([
            'eq=brightness=-0.05:saturation=0.8',  # Slightly darker and less saturated
            'vignette=PI/4'  # Subtle vignette
        ])
    
    # Adjust based on intensity (1-10)
    intensity = sentiment.get('intensity', 5)
    if intensity > 7:
        effects.append(f'vibrance={min(intensity/10, 1.5)}')  # More vibrant for intense moments
    
    # Adjust based on speaking pace
    pace = sentiment.get('pace', 'medium')
    if pace == 'fast':
        edit_point['Duration'] = f"** {float(edit_point['Duration'].split()[0]) * 0.9} seconds"  # Slightly faster
    elif pace == 'slow':
        edit_point['Duration'] = f"** {float(edit_point['Duration'].split()[0]) * 1.1} seconds"  # Slightly slower
    
    # Update technical notes with new effects
    if effects:
        tech_notes = edit_point.get('Technical Notes', '')
        if '`ffmpeg' in tech_notes:
            # Insert effects before output
            tech_notes = tech_notes.replace('output', f"-vf \"{','.join(effects)}\" output")
            edit_point['Technical Notes'] = tech_notes
    
    return edit_point

def transcribe_video(video_path):
    """Transcribe video audio to text using Whisper"""
    try:
        logger.info(f"Starting transcription for: {video_path}")
        
        # Convert video to WAV format for transcription
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(os.path.basename(video_path))[0]}.wav")
        
        # Extract audio using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono audio
            '-y',  # Overwrite output file
            wav_path
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Load audio file
        audio = open(wav_path, 'rb')
        
        # Transcribe using OpenAI's Whisper API
        logger.info("Starting Whisper transcription")
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="text"
        )
        
        # Clean up WAV file
        audio.close()
        try:
            os.remove(wav_path)
            logger.info(f"Cleaned up WAV file: {wav_path}")
        except Exception as e:
            logger.warning(f"Could not remove WAV file {wav_path}: {str(e)}")
        
        logger.info("Transcription completed successfully")
        return response
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise Exception(f"Error extracting audio: {str(e)}")
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Error during transcription: {str(e)}")

def parse_timestamp(timestamp: str) -> float:
    """
    Parse timestamp string into seconds with millisecond precision
    Handles formats:
    - HH:MM:SS.mmm
    - MM:SS.mmm
    - SS.mmm
    """
    try:
        # Remove any brackets and whitespace
        timestamp = timestamp.strip('[] \t\n\r')
        
        if not timestamp:
            return 0.0
            
        # Split into parts
        if '.' in timestamp:
            time_part, ms_part = timestamp.split('.')
        else:
            time_part, ms_part = timestamp, '0'
            
        # Split time part
        time_components = time_part.split(':')
        
        if len(time_components) == 3:  # HH:MM:SS
            h, m, s = time_components
            seconds = float(h) * 3600 + float(m) * 60 + float(s)
        elif len(time_components) == 2:  # MM:SS
            m, s = time_components
            seconds = float(m) * 60 + float(s)
        else:  # SS
            seconds = float(time_components[0])
            
        # Add milliseconds
        milliseconds = float(f"0.{ms_part}")
        
        return seconds + milliseconds
        
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing timestamp '{timestamp}': {str(e)}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error parsing timestamp '{timestamp}': {str(e)}")
        return 0.0

def parse_duration(duration_str: str) -> float:
    """
    Parse duration string in various formats
    Examples:
    - "** [5.0] seconds" -> 5.0
    - "5.0 seconds" -> 5.0
    - "[5.0]" -> 5.0
    - "5.0" -> 5.0
    """
    try:
        # Remove all non-numeric characters except decimal point
        duration_str = duration_str.replace('**', '').replace('[', '').replace(']', '')
        duration_str = duration_str.replace('seconds', '').strip()
        return float(duration_str)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing duration '{duration_str}': {str(e)}")
        return 0.0

def apply_edit_instructions(video_paths: List[str], edit_points: List[Dict], output_folder: str) -> str:
    try:
        logger.info("Starting video processing...")
        processed_segments = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each video and its edit points
            for video_idx, (video_path, video_edits) in enumerate(zip(video_paths, edit_points)):
                # Remove duplicate edit points by using a set of timestamps
                seen_timestamps = set()
                unique_edits = []
                
                for edit in video_edits:
                    timestamp = edit.get('Timestamp', '')
                    if timestamp and timestamp not in seen_timestamps:
                        seen_timestamps.add(timestamp)
                        unique_edits.append(edit)
                
                # Sort edits by timestamp
                unique_edits.sort(key=lambda x: parse_timestamp(x.get('Timestamp', '0')))
                
                # Process each unique edit point
                for edit_idx, edit in enumerate(unique_edits):
                    try:
                        # Extract timing information
                        start_time = edit.get('Timestamp', '00:00:00.000')
                        end_time = edit.get('EndTime', '')
                        text = edit.get('Transcription', '')
                        
                        # Parse timestamps
                        start_seconds = parse_timestamp(start_time)
                        duration_str = edit.get('Duration', '0')
                        duration = parse_duration(duration_str)
                        
                        if duration <= 0:
                            logger.warning(f"Invalid duration for segment {video_idx}_{edit_idx}: {duration}s")
                            continue
                        
                        # Calculate end time if not provided
                        end_seconds = parse_timestamp(end_time) if end_time else start_seconds + duration
                        
                        logger.debug(f"Processing segment {video_idx}_{edit_idx}: {start_seconds:.3f}s to {end_seconds:.3f}s")
                        
                        # Process segment
                        segment_output = os.path.join(temp_dir, f'segment_{video_idx}_{edit_idx}.mp4')
                        if process_segment(video_path, segment_output, start_seconds, duration, text):
                            processed_segments.append(segment_output)
                        
                    except Exception as e:
                        logger.error(f"Error processing segment {video_idx}_{edit_idx}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
            
            if not processed_segments:
                logger.error("No segments were successfully processed")
                return None
            
            # Concatenate all processed segments
            logger.info(f"Concatenating {len(processed_segments)} segments...")
            return concatenate_with_transitions(processed_segments, output_folder)
            
    except Exception as e:
        logger.error(f"Error in video editing: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_segment(video_path: str, output_path: str, start_time: float, 
                   duration: float, text: str, transition: str = 'cross_dissolve') -> bool:
    try:
        # Base FFmpeg command
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration)
        ]
        
        # Add transition effects based on type
        if transition == 'cross_dissolve':
            command.extend([
                '-vf', f'fade=t=in:st=0:d=0.5,fade=t=out:st={duration-0.5}:d=0.5'
            ])
        elif transition == 'fast_dissolve':
            command.extend([
                '-vf', f'fade=t=in:st=0:d=0.3,fade=t=out:st={duration-0.3}:d=0.3'
            ])
        elif transition == 'slow_fade':
            command.extend([
                '-vf', f'fade=t=in:st=0:d=0.8,fade=t=out:st={duration-0.8}:d=0.8'
            ])
        
        # Add output settings
        command.extend([
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-avoid_negative_ts', 'make_zero',
            output_path
        ])
        
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True)
        return True
        
    except Exception as e:
        logger.error(f"Error processing segment: {str(e)}")
        return False

def concatenate_with_transitions(segments: List[str], output_folder: str) -> str:
    """Concatenate segments with minimal transition overlap"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_folder, f'final_edited_video_{timestamp}.mp4')
        
        # Create concat file
        concat_file = os.path.join(output_folder, 'concat.txt')
        with open(concat_file, 'w') as f:
            for segment in segments:
                f.write(f"file '{segment}'\n")
        
        # Use simple concatenation to prevent transition stacking
        command = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Direct stream copy to prevent re-encoding
            output_path
        ]
        
        subprocess.run(command, check=True)
        os.remove(concat_file)
        return output_path
        
    except Exception as e:
        logger.error(f"Error concatenating segments: {str(e)}")
        return None

def generate_edit_points(transcription: str, video_duration: float) -> List[Dict]:
    try:
        logger.info("Generating edit points from transcription...")
        
        # Initialize variables
        edit_points = []
        last_timestamp = 0
        min_segment_duration = 2.0  # Minimum segment duration in seconds
        
        # Get sentiment analysis from OpenAI
        response = analyze_with_openai(transcription)
        segments = response.get('segments', [])
        
        for segment in segments:
            timestamp = parse_timestamp(segment.get('Timestamp', '0'))
            
            # Skip if too close to last edit point
            if timestamp - last_timestamp < min_segment_duration:
                continue
                
            # Calculate appropriate duration based on content
            duration = calculate_segment_duration(segment)
            
            # Ensure we don't cut off sentences
            end_of_sentence = find_sentence_end(transcription, timestamp)
            if end_of_sentence:
                duration = max(duration, end_of_sentence - timestamp)
            
            edit_point = {
                'Timestamp': format_timestamp(timestamp),
                'Duration': duration,
                'Sentiment': segment.get('Sentiment', {}),
                'Keywords': segment.get('Keywords', []),
                'Shot Type': segment.get('Shot Type', ''),
                'Transition': determine_transition(segment),
                'Transcription': extract_transcription(transcription, timestamp, timestamp + duration)
            }
            
            edit_points.append(edit_point)
            last_timestamp = timestamp
            
        return edit_points
        
    except Exception as e:
        logger.error(f"Error generating edit points: {str(e)}")
        return []

def calculate_segment_duration(segment: Dict) -> float:
    """Calculate appropriate duration based on content and sentiment"""
    base_duration = 3.0
    
    # Adjust duration based on content type
    if 'important' in segment.get('Keywords', []):
        base_duration += 1.0
    if segment.get('Sentiment', {}).get('intensity', 5) > 7:
        base_duration += 0.5
        
    return base_duration

def find_sentence_end(transcription: str, start_time: float) -> Optional[float]:
    """Find the next sentence end after start_time"""
    # Implementation to find next period, question mark, or exclamation point
    # and return its timestamp
    pass

def determine_transition(segment: Dict) -> str:
    """Determine appropriate transition based on content"""
    sentiment = segment.get('Sentiment', {})
    intensity = sentiment.get('intensity', 5)
    pace = sentiment.get('pace', 'Medium')
    
    if intensity > 7 and pace == 'Fast':
        return 'fast_dissolve'
    elif intensity < 4:
        return 'slow_fade'
    else:
        return 'cross_dissolve'

def extract_transcription(transcription: str, start_time: float, end_time: float) -> str:
    """Extract transcription between timestamps while preserving complete sentences"""
    try:
        # Find the complete sentence that contains start_time
        sentence_start = find_sentence_start(transcription, start_time)
        sentence_end = find_sentence_end(transcription, end_time)
        
        if sentence_start is None or sentence_end is None:
            return ""
            
        return transcription[sentence_start:sentence_end].strip()
        
    except Exception as e:
        logger.error(f"Error extracting transcription: {str(e)}")
        return ""

def find_sentence_start(transcription: str, timestamp: float) -> Optional[int]:
    """Find the start of the sentence containing timestamp"""
    # Implementation to find previous sentence boundary
    # This should look for periods, question marks, exclamation points
    # and return the position after them
    pass

def analyze_with_openai(transcription: str) -> Dict:
    """Enhanced OpenAI analysis focusing on story continuity"""
    try:
        response = openai.ChatCompletion.create(
            model="whisper-1",
            messages=[{
                "role": "system",
                "content": """Analyze the video transcription and identify key story points. 
                For each segment, indicate:

                1. IMPORTANCE: Rate 1-10 (10 being crucial plot points that must be kept)
                2. CONTEXT: Is this part of an ongoing sentence/thought? (yes/no)
                3. CONTINUITY: Does this connect directly to the next segment? (yes/no)

                Format each segment as:

                [Timestamp]
                Importance: (1-10)
                Context: (Continuing/Complete)
                Continuity: (Connects/Standalone)
                Sentiment: {Positive/Negative/Neutral}, Intensity: 1-10
                Keywords: key story elements
                Shot Type: {Wide/Medium/Close-up}
                Transition: {subtle_fade/match_cut/none}
                Duration: X.X seconds (minimum needed for comprehension)

                RULES:
                - Mark all story-critical segments (plot points) as Importance 8+
                - Keep segments with ongoing dialogue together
                - Minimum duration should allow for complete thoughts
                - Only suggest transitions between complete thoughts
                - Preserve context between related segments"""
            }, {
                "role": "user",
                "content": transcription
            }]
        )
        
        return parse_openai_response(response)
        
    except Exception as e:
        logger.error(f"Error in OpenAI analysis: {str(e)}")
        return {'segments': []}

def parse_openai_response(response: Dict) -> Dict:
    """Parse OpenAI's response into structured edit points with segments"""
    try:
        # Extract the content from OpenAI's response
        content = response['choices'][0]['message']['content']
        
        # Initialize the result structure
        result = {
            'segments': []
        }
        
        # Split the content into lines and parse each segment
        lines = content.strip().split('\n')
        current_segment = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Start a new segment when timestamp is found
            if '[' in line and ']' in line and ':' in line:
                if current_segment:
                    result['segments'].append(current_segment)
                current_segment = {
                    'Timestamp': extract_timestamp(line),
                    'Sentiment': {'tone': '', 'intensity': 5, 'pace': 'Medium'},
                    'Keywords': [],
                    'Shot Type': '',
                    'Transition': 'none',
                    'Duration': 3.0  # Default duration
                }
            
            # Parse different types of information
            if 'Sentiment:' in line or 'SENTIMENT:' in line:
                current_segment['Sentiment'] = parse_sentiment(line)
            elif 'Keywords:' in line or 'KEYWORDS:' in line:
                current_segment['Keywords'] = parse_keywords(line)
            elif 'Shot Type:' in line or 'SHOT:' in line:
                current_segment['Shot Type'] = parse_shot_type(line)
            elif 'Transition:' in line:
                current_segment['Transition'] = parse_transition(line)
            elif 'Duration:' in line:
                current_segment['Duration'] = parse_duration(line)
        
        # Add the last segment
        if current_segment:
            result['segments'].append(current_segment)
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {str(e)}")
        logger.error(f"Response content: {response}")
        return {'segments': []}

def extract_timestamp(line: str) -> str:
    """Extract timestamp from line"""
    try:
        # Find the timestamp pattern [HH:MM:SS.mmm]
        import re
        match = re.search(r'\[(\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\]', line)
        if match:
            return match.group(1)
        return "00:00:00.000"
    except Exception as e:
        logger.error(f"Error extracting timestamp: {str(e)}")
        return "00:00:00.000"

def parse_sentiment(line: str) -> Dict:
    """Parse sentiment information"""
    try:
        sentiment = {
            'tone': 'Neutral',
            'intensity': 5,
            'pace': 'Medium'
        }
        
        # Extract tone
        if 'positive' in line.lower():
            sentiment['tone'] = 'Positive'
        elif 'negative' in line.lower():
            sentiment['tone'] = 'Negative'
        
        # Extract intensity (1-10)
        intensity_match = re.search(r'intensity[:\s]+(\d+)', line.lower())
        if intensity_match:
            sentiment['intensity'] = int(intensity_match.group(1))
        
        # Extract pace
        if 'fast' in line.lower():
            sentiment['pace'] = 'Fast'
        elif 'slow' in line.lower():
            sentiment['pace'] = 'Slow'
        
        return sentiment
    except Exception as e:
        logger.error(f"Error parsing sentiment: {str(e)}")
        return {'tone': 'Neutral', 'intensity': 5, 'pace': 'Medium'}

def parse_keywords(line: str) -> List[str]:
    """Parse keywords from line"""
    try:
        # Remove prefix and extract words
        keywords_str = line.split(':', 1)[1] if ':' in line else line
        # Clean up the keywords and split them
        keywords = [k.strip(' "[]\'') for k in keywords_str.split(',')]
        return [k for k in keywords if k]
    except Exception as e:
        logger.error(f"Error parsing keywords: {str(e)}")
        return []

def parse_shot_type(line: str) -> str:
    """Parse shot type from line"""
    try:
        return line.split(':', 1)[1].strip(' "[]\'') if ':' in line else ''
    except Exception as e:
        logger.error(f"Error parsing shot type: {str(e)}")
        return ''

def parse_transition(line: str) -> str:
    """Parse transition type from line"""
    try:
        transition = line.split(':', 1)[1].strip().lower() if ':' in line else ''
        # Map transition descriptions to our supported types
        if 'fast' in transition or 'quick' in transition:
            return 'fast_dissolve'
        elif 'slow' in transition or 'gradual' in transition:
            return 'slow_fade'
        return 'cross_dissolve'
    except Exception as e:
        logger.error(f"Error parsing transition: {str(e)}")
        return 'cross_dissolve'

def process_video_with_edits(video_path: str, edit_points: List[Dict], output_folder: str) -> Tuple[str, str]:
    """Process video and generate edit log"""
    try:
        edit_log = []
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_segments = []
            total_segments = len(edit_points)
            
            # First pass: identify story-critical segments
            for i, edit in enumerate(edit_points):
                importance = int(edit.get('Importance', 5))
                timestamp = edit.get('Timestamp', '00:00:00.000')
                duration = float(edit.get('Duration', 3.0))
                
                # Always include high-importance segments and their context
                should_include = (
                    importance >= 6 or  # Include moderately important content
                    edit.get('Context') == 'Continuing' or  # Include ongoing dialogue
                    edit.get('Continuity') == 'Connects'  # Include connected content
                )
                
                if should_include:
                    segment_output = os.path.join(temp_dir, f'segment_{i:03d}.mp4')
                    metadata = process_enhanced_segment(
                        video_path=video_path,
                        output_path=segment_output,
                        start_time=parse_timestamp(timestamp),
                        duration=duration,
                        sentiment=edit.get('Sentiment', {}),
                        transition=edit.get('Transition', 'none'),
                        is_first=(i == 0),
                        is_last=(i == total_segments - 1),
                        importance=importance
                    )
                    
                    if metadata['success']:
                        processed_segments.append(segment_output)
                        edit_log.append({
                            'segment_number': i + 1,
                            'timestamp': timestamp,
                            'duration': duration,
                            'importance': importance,
                            'transition': metadata['transition_type'],
                            'sentiment': metadata.get('sentiment', {}),
                            'keywords': edit.get('Keywords', [])
                        })
            
            if not processed_segments:
                raise ValueError("No segments were successfully processed")
            
            # Generate detailed edit log
            log_path = generate_edit_log(edit_log, output_folder)
            
            # Concatenate segments
            final_video = concatenate_with_transitions(processed_segments, output_folder)
            
            return final_video, log_path
            
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        return None, None

def generate_edit_log(edit_log: List[Dict], output_folder: str) -> str:
    """Generate detailed edit log file"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output_folder, f'edit_log_{timestamp}.txt')
        
        with open(log_path, 'w') as f:
            f.write("FINAL EDIT LOG\n")
            f.write("==============\n\n")
            
            total_duration = 0
            for entry in edit_log:
                total_duration += entry['duration']
                f.write(f"Segment {entry['segment_number']}\n")
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write(f"Duration: {entry['duration']:.2f} seconds\n")
                f.write(f"Importance: {entry['importance']}/10\n")
                f.write(f"Transition: {entry['transition']}\n")
                if entry['sentiment']:
                    f.write(f"Sentiment: {entry['sentiment'].get('tone', 'Unknown')}, "
                           f"Intensity: {entry['sentiment'].get('intensity', 'Unknown')}\n")
                if entry['keywords']:
                    f.write(f"Keywords: {', '.join(entry['keywords'])}\n")
                f.write("\n")
            
            f.write(f"\nTotal Duration: {total_duration:.2f} seconds\n")
            
        return log_path
        
    except Exception as e:
        logger.error(f"Error generating edit log: {str(e)}")
        return None

# Define metadata type structure
SegmentMetadata = TypedDict('SegmentMetadata', {
    'start_time': float,
    'duration': float,
    'transition_type': str,
    'importance': int,
    'sentiment': Dict[str, Any],
    'success': bool,
    'cut_type': str,
    'scene_type': str,
    'audio_fade': Optional[float],
    'video_fade': Optional[float]
})

def process_enhanced_segment(
    video_path: str, 
    output_path: str,
    start_time: float,
    duration: float,
    sentiment: Dict,
    transition: str,
    is_first: bool,
    is_last: bool,
    importance: int = 5,
    params: Dict = None
) -> SegmentMetadata:
    """
    Process video segment with configurable parameters
    
    Parameters:
    - video_path: Source video file
    - output_path: Where to save processed segment
    - start_time: Start timestamp in seconds
    - duration: Segment duration in seconds
    - sentiment: Sentiment analysis dict
    - transition: Type of transition requested
    - is_first: Is this the first segment?
    - is_last: Is this the last segment?
    - importance: Segment importance (1-10)
    - params: Optional parameter overrides
    """
    
    # Default parameters
    default_params = {
        'fade_duration': 0.08,  # 80ms default fade
        'fade_threshold': 8,    # Importance level above which fades are disabled
        'min_duration': 1.0,    # Minimum segment duration
        'audio_fade': 0.04,     # 40ms audio fade
        'transition_types': {
            'scene_change': 'quick_fade',
            'continuous': 'none',
            'dramatic': 'cross_fade'
        },
        'video_quality': {
            'crf': 23,
            'preset': 'medium'
        }
    }
    
    # Override defaults with provided params
    params = {**default_params, **(params or {})}
    
    try:
        filters = []
        audio_filters = []
        
        metadata: SegmentMetadata = {
            'start_time': start_time,
            'duration': duration,
            'transition_type': 'none',
            'importance': importance,
            'sentiment': sentiment,
            'success': False,
            'cut_type': 'straight_cut',
            'scene_type': 'continuous' if importance >= params['fade_threshold'] else 'scene_change',
            'audio_fade': None,
            'video_fade': None
        }
        
        # Determine transition type
        if not is_first and not is_last and importance < params['fade_threshold']:
            if transition == 'dramatic':
                fade_duration = params['fade_duration'] * 1.5
                filters.append(f'fade=t=in:st=0:d={fade_duration},fade=t=out:st={duration-fade_duration}:d={fade_duration}')
                metadata['transition_type'] = 'dramatic_fade'
                metadata['video_fade'] = fade_duration
            elif transition == 'scene_change':
                filters.append(f'fade=t=in:st=0:d={params["fade_duration"]},fade=t=out:st={duration-params["fade_duration"]}:d={params["fade_duration"]}')
                metadata['transition_type'] = 'quick_fade'
                metadata['video_fade'] = params['fade_duration']
            
            # Add audio fades
            audio_filters.append(f'afade=t=in:st=0:d={params["audio_fade"]},afade=t=out:st={duration-params["audio_fade"]}:d={params["audio_fade"]}')
            metadata['audio_fade'] = params['audio_fade']
        
        # Construct FFmpeg command
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration)
        ]
        
        # Add video filters
        if filters:
            command.extend(['-vf', ','.join(filters)])
        
        # Add audio filters
        if audio_filters:
            command.extend(['-af', ','.join(audio_filters)])
        
        # Add output settings
        command.extend([
            '-c:v', 'libx264',
            '-preset', params['video_quality']['preset'],
            '-crf', str(params['video_quality']['crf']),
            '-c:a', 'aac',
            '-b:a', '128k',
            '-avoid_negative_ts', 'make_zero',
            output_path
        ])
        
        logger.debug(f"Processing segment: {start_time:.2f}s to {start_time + duration:.2f}s")
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        subprocess.run(command, check=True)
        metadata['success'] = True
        return metadata
        
    except Exception as e:
        logger.error(f"Error processing segment: {str(e)}")
        return metadata

def generate_enhanced_edit_log(edit_log: List[Dict], output_folder: str) -> str:
    """Generate detailed edit log with timecode and transition info"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output_folder, f'edit_log_{timestamp}.txt')
        
        with open(log_path, 'w') as f:
            f.write("FINAL EDIT LOG\n")
            f.write("==============\n\n")
            
            total_duration = 0
            for entry in edit_log:
                total_duration += entry['duration']
                timecode = format_timecode(entry['timestamp'])
                
                f.write(f"Segment {entry['segment_number']:03d}\n")
                f.write(f"{'='*40}\n")
                f.write(f"Timecode: {timecode}\n")
                f.write(f"Duration: {entry['duration']:.2f}s\n")
                f.write(f"Cut Type: {entry.get('cut_type', 'straight_cut')}\n")
                
                if entry.get('video_fade'):
                    f.write(f"Video Fade: {entry['video_fade']:.2f}s\n")
                if entry.get('audio_fade'):
                    f.write(f"Audio Fade: {entry['audio_fade']*1000:.0f}ms\n")
                
                f.write(f"Importance: {entry['importance']}/10\n")
                
                if entry['sentiment']:
                    f.write("Sentiment:\n")
                    f.write(f"  - Tone: {entry['sentiment'].get('tone', 'Unknown')}\n")
                    f.write(f"  - Intensity: {entry['sentiment'].get('intensity', 'Unknown')}\n")
                    f.write(f"  - Pace: {entry['sentiment'].get('pace', 'Unknown')}\n")
                
                if entry['keywords']:
                    f.write(f"Keywords: {', '.join(entry['keywords'])}\n")
                
                f.write("\n")
            
            f.write("\nSUMMARY\n")
            f.write("=======\n")
            f.write(f"Total Duration: {total_duration:.2f}s\n")
            f.write(f"Total Segments: {len(edit_log)}\n")
        
        return log_path
        
    except Exception as e:
        logger.error(f"Error generating edit log: {str(e)}")
        return None

def format_timecode(timestamp: str) -> str:
    """Convert timestamp to readable timecode"""
    try:
        time_parts = timestamp.strip('[]').split(':')
        return f"{time_parts[0]}:{time_parts[1]}:{time_parts[2]}"
    except:
        return timestamp

def analyze_segments(video_path: str, transcription: str) -> List[Dict]:
    """Analyze video segments and create a detailed transition plan"""
    
    segments = []
    try:
        # First, split transcription into sentences and get their timestamps
        sentences = split_into_sentences(transcription)
        
        for i, sentence in enumerate(sentences):
            # Get timing info
            start_time = extract_timestamp(sentence['start'])
            end_time = extract_timestamp(sentence['end'])
            
            # Analyze the content
            segment = {
                'index': i,
                'text': sentence['text'],
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'is_question': '?' in sentence['text'],
                'is_statement_end': sentence['text'].strip().endswith('.'),
                'connects_to_next': not sentence['text'].strip().endswith('.'),
                'sentiment': analyze_sentence_sentiment(sentence['text'])
            }
            
            # Determine transition type
            if i > 0:
                prev_segment = segments[-1]
                segment['transition'] = determine_transition(prev_segment, segment)
            else:
                segment['transition'] = 'none'
            
            segments.append(segment)
            
        # Log the analysis
        logger.info("Segment Analysis:")
        for segment in segments:
            logger.info(f"""
Segment {segment['index']}:
Time: {format_timestamp(segment['start_time'])} -> {format_timestamp(segment['end_time'])}
Text: {segment['text']}
Duration: {segment['duration']:.2f}s
Transition: {segment['transition']}
Connects to next: {segment['connects_to_next']}
---""")
        
        return segments
        
    except Exception as e:
        logger.error(f"Error analyzing segments: {str(e)}")
        return []

def determine_transition(prev_segment: Dict, current_segment: Dict) -> str:
    """Determine appropriate transition type between segments"""
    
    # No transition if segments are connected
    if prev_segment['connects_to_next']:
        return 'none'
        
    # Time gap between segments
    time_gap = current_segment['start_time'] - prev_segment['end_time']
    
    # If there's a significant time gap
    if time_gap > 0.5:
        return 'fade'
        
    # If it's a natural break (end of statement)
    if prev_segment['is_statement_end']:
        return 'quick_dissolve'
        
    # If it's a dramatic shift in sentiment
    if (prev_segment['sentiment']['tone'] != 
        current_segment['sentiment']['tone']):
        return 'cross_dissolve'
        
    # Default to no transition
    return 'none'

def split_into_sentences(transcription: str) -> List[Dict]:
    """Split transcription into sentence segments with timing"""
    try:
        # Implementation depends on your transcription format
        # This is a placeholder - adjust based on your actual format
        sentences = []
        # ... split logic ...
        return sentences
    except Exception as e:
        logger.error(f"Error splitting transcription: {str(e)}")
        return []

def analyze_sentence_sentiment(text: str) -> Dict:
    """Analyze sentiment of a sentence without using external APIs"""
    try:
        # Initialize sentiment structure
        sentiment = {
            'tone': 'neutral',
            'intensity': 5,
            'pace': 'medium',
            'is_question': False,
            'is_exclamation': False
        }
        
        # Handle empty or invalid text
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {text}")
            return sentiment
            
        # Basic sentiment indicators
        positive_words = {'good', 'great', 'awesome', 'amazing', 'love', 'happy', 
                         'excited', 'best', 'beautiful', 'fun', 'nice', 'cool'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 
                         'worst', 'horrible', 'boring', 'wrong', 'difficult'}
        intensity_words = {'very', 'really', 'extremely', 'absolutely', 'totally', 
                         'completely', 'super', 'so', 'definitely'}
        
        # Convert to lowercase and split into words
        text_lower = text.lower().strip()
        words = [word.strip('.,!?') for word in text_lower.split()]
        
        # Count sentiment indicators
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        intensity_modifiers = sum(1 for word in words if word in intensity_words)
        
        # Determine base tone
        if positive_count > negative_count:
            sentiment['tone'] = 'positive'
        elif negative_count > positive_count:
            sentiment['tone'] = 'negative'
            
        # Adjust intensity (base 5, modified by indicators)
        base_intensity = 5
        intensity_shift = (positive_count + negative_count) * 2
        intensity_multiplier = 1 + (0.5 * intensity_modifiers)
        sentiment['intensity'] = min(10, max(1, int(
            (base_intensity + intensity_shift) * intensity_multiplier
        )))
        
        # Calculate words per second (assuming average speaking pace)
        # Average English speaker: ~150 words per minute = 2.5 words per second
        text_duration = len(words) / 2.5  # estimated duration in seconds
        words_per_second = len(words) / max(1, text_duration)
        
        # Determine pace
        if words_per_second > 3:
            sentiment['pace'] = 'fast'
        elif words_per_second < 2:
            sentiment['pace'] = 'slow'
            
        # Check for questions and exclamations
        sentiment['is_question'] = '?' in text
        sentiment['is_exclamation'] = '!' in text
        
        logger.debug(f"Sentiment analysis for '{text[:50]}...': {sentiment}")
        return sentiment
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {
            'tone': 'neutral',
            'intensity': 5,
            'pace': 'medium',
            'is_question': False,
            'is_exclamation': False
        }

def determine_transition(prev_segment: Dict, current_segment: Dict) -> Tuple[str, float]:
    """Determine transition type and duration based on sentiment analysis"""
    try:
        # Get sentiment info
        prev_sentiment = prev_segment.get('sentiment', {})
        curr_sentiment = current_segment.get('sentiment', {})
        
        # Default transition values
        transition_type = 'none'
        transition_duration = 0.0
        
        # Check for connected speech
        if prev_segment.get('connects_to_next', False):
            return 'none', 0.0
            
        # Dramatic sentiment changes get cross dissolves
        if (prev_sentiment.get('tone') != curr_sentiment.get('tone') and 
           abs(prev_sentiment.get('intensity', 5) - curr_sentiment.get('intensity', 5)) > 3):
            return 'cross_dissolve', 0.15
            
        # Questions get quick fades
        if prev_sentiment.get('is_question'):
            return 'quick_fade', 0.08
            
        # Exclamations get fast dissolves
        if prev_sentiment.get('is_exclamation'):
            return 'fast_dissolve', 0.12
            
        # Pace changes affect transition duration
        if prev_sentiment.get('pace') != curr_sentiment.get('pace'):
            if curr_sentiment.get('pace') == 'fast':
                return 'quick_cut', 0.05
            else:
                return 'smooth_dissolve', 0.2
        
        # Default to subtle transition for natural breaks
        return 'subtle_fade', 0.1
        
    except Exception as e:
        logger.error(f"Error determining transition: {str(e)}")
        return 'none', 0.0

def process_enhanced_segment(video_path: str, output_path: str, 
                           start_time: float, duration: float,
                           transition_type: str, transition_duration: float,
                           is_first: bool, is_last: bool) -> bool:
    """Process video segment with specified transition"""
    try:
        filters = []
        
        # Apply transition based on type
        if not is_first and transition_type != 'none':
            if transition_type in ['quick_fade', 'subtle_fade']:
                filters.append(f'fade=t=in:st=0:d={transition_duration}')
            elif transition_type in ['cross_dissolve', 'fast_dissolve', 'smooth_dissolve']:
                filters.append(f'fade=t=in:st=0:d={transition_duration}:alpha=1')
                
        if not is_last and transition_type != 'none':
            if transition_type in ['quick_fade', 'subtle_fade']:
                filters.append(f'fade=t=out:st={duration-transition_duration}:d={transition_duration}')
            elif transition_type in ['cross_dissolve', 'fast_dissolve', 'smooth_dissolve']:
                filters.append(f'fade=t=out:st={duration-transition_duration}:d={transition_duration}:alpha=1')
        
        # Construct FFmpeg command
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration)
        ]
        
        if filters:
            command.extend(['-vf', ','.join(filters)])
        
        command.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-avoid_negative_ts', 'make_zero',
            output_path
        ])
        
        logger.debug(f"Processing segment with transition: {transition_type} ({transition_duration:.3f}s)")
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        subprocess.run(command, check=True)
        return True
        
    except Exception as e:
        logger.error(f"Error processing segment: {str(e)}")
        return False

def analyze_video_segments(transcription: str, video_metadata: Dict) -> List[Dict]:
    """Generate clean, non-redundant edit points from transcription"""
    try:
        segments = []
        current_time = 0.0
        
        # Split into natural speech segments
        sentences = split_into_sentences(transcription)
        if not sentences:
            logger.error("No sentences found in transcription")
            return []
            
        for i, sentence in enumerate(sentences):
            try:
                # Validate sentence structure
                if not isinstance(sentence, dict) or 'text' not in sentence:
                    logger.error(f"Invalid sentence structure at index {i}")
                    continue
                    
                # Calculate proper duration based on word count and pace
                text = sentence['text'].strip()
                word_count = len(text.split())
                estimated_duration = max(1.0, word_count * 0.3)  # minimum 1 second
                
                segment = {
                    'index': i,
                    'start_time': current_time,
                    'duration': estimated_duration,
                    'text': text,
                    'scene_type': detect_scene_type(text),
                    'is_question': '?' in text,
                    'connects_to_next': detect_continuation(
                        text,
                        sentences[i+1]['text'] if i < len(sentences)-1 else None
                    ),
                    'sentiment': analyze_sentence_sentiment(text)
                }
                
                # Validate segment timing
                if segment['start_time'] < 0 or segment['duration'] <= 0:
                    logger.error(f"Invalid timing for segment {i}")
                    continue
                    
                segments.append(segment)
                current_time += estimated_duration
                
                logger.debug(f"Created segment {i}: {segment['scene_type']} "
                           f"({segment['duration']:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error processing segment {i}: {str(e)}")
                continue
        
        return segments
        
    except Exception as e:
        logger.error(f"Error in video segment analysis: {str(e)}")
        return []

def detect_scene_type(text: str) -> str:
    """Detect the type of scene based on text content"""
    text_lower = text.lower()
    
    scene_indicators = {
        'introduction': {'hey', 'hi', "what's up", 'hello', 'welcome'},
        'action': {'going', 'doing', 'making', 'trying', 'let\'s'},
        'transition': {'now', 'next', 'then', 'after that', 'moving on'},
        'conclusion': {'thanks', 'thank you', 'bye', 'see you', 'that\'s all'}
    }
    
    for scene_type, indicators in scene_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            return scene_type
    
    return 'content'  # default scene type

def detect_continuation(current_text: str, next_text: Optional[str]) -> bool:
    """Detect if the current text continues into the next segment"""
    if not next_text:
        return False
        
    # Check for incomplete sentences
    if not any(current_text.strip().endswith(end) for end in {'.', '!', '?'}):
        return True
        
    # Check for connecting words at start of next sentence
    connecting_words = {'and', 'but', 'or', 'so', 'because', 'then'}
    next_starts_with_connector = any(
        next_text.lower().strip().startswith(word) 
        for word in connecting_words
    )
    
    return next_starts_with_connector

def generate_clean_edit_log(segments: List[Dict], output_path: str):
    """Generate non-redundant edit log"""
    with open(output_path, 'w') as f:
        f.write("CLEAN EDIT LOG\n")
        f.write("=============\n\n")
        
        for segment in segments:
            f.write(f"Segment {segment['index']:03d}\n")
            f.write(f"Timestamp: {format_timestamp(segment['start_time'])}\n")
            f.write(f"Duration: {segment['duration']:.2f}s\n")
            f.write(f"Text: {segment['text']}\n")
            f.write(f"Scene Type: {segment['scene_type']}\n")
            f.write(f"Transition: {segment['transition']['type']} "
                   f"({segment['transition']['duration']*1000:.0f}ms)\n")
            f.write("\n")

def process_video_segment(input_path: str, output_path: str, segment: Dict) -> bool:
    """Process single video segment with clean cuts"""
    try:
        command = [
            'ffmpeg', '-y',
            '-ss', str(segment['timestamp']),
            '-i', input_path,
            '-t', str(segment['duration']),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-af', 'volume=1.5',
            '-avoid_negative_ts', 'make_zero',
            '-async', '1',  # Help maintain audio sync
            output_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error processing segment: {str(e)}")
        return False

def process_crossfade_segment(input_path: str, output_path: str, segment: Dict) -> bool:
    """Handle cross dissolve transitions properly"""
    try:
        xfade_duration = min(0.5, segment['duration'] * 0.2)
        
        command = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-filter_complex',
            f"[0:v]trim=start={segment['start_time']}:duration={segment['duration']},"
            f"setpts=PTS-STARTPTS[v1];"
            f"[0:v]trim=start={segment['start_time']+segment['duration']-xfade_duration}:"
            f"duration={xfade_duration},setpts=PTS-STARTPTS[v2];"
            f"[v1][v2]xfade=duration={xfade_duration}:offset={segment['duration']-xfade_duration}",
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac',
            output_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error in crossfade: {str(e)}")
        return False

def generate_edit_instructions(video_info: Dict) -> List[Dict]:
    """Generate streamlined edit instructions based on video content and style"""
    
    # Base structure for edit points
    edit_points = []
    
    # Get video metadata
    duration = video_info['duration']
    transcript = video_info['transcript']
    style = video_info['style']
    sentiment = video_info['sentiment_analysis']
    
    
    
def determine_shot_type(content: str, preset: Dict) -> str:
    """Determine appropriate shot type based on content and style preset"""
    if "show" in content.lower():
        return preset['shot_types']['showcase']
    elif "reaction" in content.lower():
        return preset['shot_types']['reaction']
    # ... etc

def select_transition(sentiment: float, preset: Dict) -> str:
    """Select transition based on emotional intensity and style"""
    if sentiment > 0.7:  # High energy
        return preset['transitions']['high_energy']
    elif sentiment < 0.3:  # Low energy
        return preset['transitions']['low_energy']
    return preset['transitions']['default']

class PacingStyle(Enum):
    RAPID = "rapid"
    BALANCED = "balanced"
    GRADUAL = "gradual"

@dataclass
class ContentSegment:
    text: str
    start_time: float
    duration: float
    energy_level: float
    speaking_pace: float
    key_moments: bool

@dataclass
class EditingStyle:
    base_style: str
    pacing: PacingStyle
    transition_preferences: List[str]
    color_grading: Dict[str, float]
    effects_intensity: float

def analyze_content_pacing(transcript: List[Dict]) -> PacingStyle:
    """Determine natural pacing from transcript"""
    # Analyze speaking patterns and content density
    words_per_second = []
    for segment in transcript:
        word_count = len(segment['text'].split())
        words_per_second.append(word_count / segment['duration'])
    
    avg_pace = np.mean(words_per_second)
    
    if avg_pace > 2.5:  # Fast-paced speech
        return PacingStyle.RAPID
    elif avg_pace < 1.5:  # Slower, deliberate speech
        return PacingStyle.GRADUAL
    return PacingStyle.BALANCED

def generate_dynamic_style(base_style: str, transcript: List[Dict]) -> EditingStyle:
    """Generate unique editing style based on content"""
    
    # Analyze content characteristics
    pacing = analyze_content_pacing(transcript)
    energy_levels = analyze_energy_levels(transcript)
    
    # Create variation in effects intensity
    base_intensity = 0.7
    random_variation = np.random.uniform(-0.2, 0.2)
    effects_intensity = max(0.3, min(1.0, base_intensity + random_variation))
    
    # Dynamic color grading based on content mood
    color_grading = {
        'vibrance': 0.8 + (energy_levels['avg'] * 0.2),
        'contrast': 1.0 + (energy_levels['variance'] * 0.3),
        'saturation': 0.9 + (energy_levels['max'] * 0.1)
    }
    
    return EditingStyle(
        base_style=base_style,
        pacing=pacing,
        transition_preferences=generate_transition_list(energy_levels),
        color_grading=color_grading,
        effects_intensity=effects_intensity
    )
def calculate_energy_score(text: str) -> float:
    """
    Calculate energy score of text based on multiple factors:
    - Exclamation marks
    - ALL CAPS words
    - High energy keywords
    - Punctuation density
    - Word choice
    """
    
    HIGH_ENERGY_KEYWORDS = {
        'amazing', 'awesome', 'incredible', 'exciting', 'wow', 
        'fantastic', 'brilliant', 'super', 'perfect', 'love',
        'great', 'best', 'beautiful', 'excellent', 'wonderful'
    }
    
    INTENSITY_MARKERS = {
        'very', 'really', 'so', 'totally', 'absolutely',
        'definitely', 'extremely', 'incredibly', 'super'
    }
    
    # Initialize base score
    score = 0.5  # Start at neutral
    
    # Normalize text
    text = text.strip().lower()
    
    # Count exclamation marks (0.1 per exclamation, max 0.3)
    exclamation_score = min(text.count('!') * 0.1, 0.3)
    score += exclamation_score
    
    # Count words in ALL CAPS in original text
    caps_words = sum(1 for word in text.split() if word.isupper())
    caps_score = min(caps_words * 0.05, 0.2)
    score += caps_score
    
    # Check for high energy keywords
    words = set(text.lower().split())
    keyword_matches = words.intersection(HIGH_ENERGY_KEYWORDS)
    intensity_matches = words.intersection(INTENSITY_MARKERS)
    
    # Add keyword scores
    keyword_score = min(len(keyword_matches) * 0.1, 0.3)
    intensity_score = min(len(intensity_matches) * 0.05, 0.2)
    score += keyword_score + intensity_score
    
    # Check punctuation density (?, !)
    total_chars = len(text)
    if total_chars > 0:
        punctuation_count = sum(1 for char in text if char in '!?')
        punctuation_density = punctuation_count / total_chars
        score += min(punctuation_density * 2, 0.2)
    
    # Normalize final score to 0-1 range
    return min(max(score, 0.0), 1.0)

def analyze_energy_levels(transcript: List[Dict]) -> Dict[str, float]:
    """Analyze energy levels in content"""
    energy_scores = []
    for segment in transcript:
        # Score based on keywords, punctuation, and sentiment
        score = calculate_energy_score(segment['text'])
        energy_scores.append(score)
    
    return {
        'avg': np.mean(energy_scores),
        'max': np.max(energy_scores),
        'variance': np.var(energy_scores)
    }

def generate_transition_list(energy_levels: Dict[str, float]) -> List[str]:
    """Generate varied transition list based on energy"""
    transitions = []
    if energy_levels['avg'] > 0.7:
        transitions.extend(['quick_cut', 'zoom_blur', 'flash'])
    elif energy_levels['avg'] < 0.4:
        transitions.extend(['dissolve', 'fade', 'smooth_slide'])
    else:
        transitions.extend(['cut', 'crossfade', 'push'])
    
    # Add some randomization
    np.random.shuffle(transitions)
    return transitions

@dataclass
class EditPoint:
    timestamp: float
    duration: float
    type: str
    transition: str
    energy_score: float
    importance_score: float

def generate_strategic_edit_points(combined_transcript: List[Dict], total_duration: float) -> List[EditPoint]:
    """Generate key edit points from combined transcript analysis"""
    
    # Aim for 8-12 strategic edits for a typical video
    MIN_EDITS = 8
    MAX_EDITS = 12
    MIN_SEGMENT_DURATION = 2.0  # Minimum seconds per segment
    
    key_moments = []
    
    # 1. Find high energy moments
    for segment in combined_transcript:
        energy_score = calculate_energy_score(segment['text'])
        importance_score = calculate_importance_score(segment['text'])
        
        if energy_score > 0.7 or importance_score > 0.7:
            key_moments.append({
                'timestamp': segment['start_time'],
                'duration': segment['duration'],
                'energy': energy_score,
                'importance': importance_score,
                'text': segment['text']
            })
    
    # 2. Select best edit points
    selected_points = []
    
    # Always include opening
    if key_moments:
        opening = key_moments[0]
        selected_points.append(EditPoint(
            timestamp=opening['timestamp'],
            duration=min(3.0, opening['duration']),
            type='opening',
            transition='fade_in',
            energy_score=opening['energy'],
            importance_score=opening['importance']
        ))
    
    # Select dramatic moments for middle edits
    middle_points = sorted(
        key_moments[1:-1],
        key=lambda x: (x['energy'] + x['importance']) / 2,
        reverse=True
    )
    
    # Calculate how many middle points we need
    target_edits = min(
        MAX_EDITS,
        max(MIN_EDITS, total_duration // MIN_SEGMENT_DURATION)
    )
    
    for point in middle_points[:target_edits-2]:  # -2 for opening/closing
        transition = select_transition(point['energy'])
        selected_points.append(EditPoint(
            timestamp=point['timestamp'],
            duration=min(4.0, point['duration']),
            type='highlight',
            transition=transition,
            energy_score=point['energy'],
            importance_score=point['importance']
        ))
    
    # Always include closing
    if key_moments:
        closing = key_moments[-1]
        selected_points.append(EditPoint(
            timestamp=closing['timestamp'],
            duration=min(3.0, closing['duration']),
            type='closing',
            transition='fade_out',
            energy_score=closing['energy'],
            importance_score=closing['importance']
        ))
    
    return selected_points

def select_transition(energy_score: float) -> str:
    """Select appropriate transition based on energy level"""
    if energy_score > 0.8:
        return np.random.choice(['quick_cut', 'zoom_blur', 'flash'])
    elif energy_score > 0.5:
        return np.random.choice(['cut', 'crossfade', 'push'])
    else:
        return np.random.choice(['dissolve', 'fade', 'smooth_slide'])

def calculate_importance_score(text: str) -> float:
    """Calculate importance of a segment based on content markers"""
    IMPORTANT_MARKERS = {
        'key', 'important', 'main', 'crucial', 'essential',
        'remember', 'note', 'highlight', 'focus', 'critical',
        'finally', 'conclusion', 'therefore', 'result', 'so',
        'introducing', 'presenting', 'showing', 'demonstrating'
    }
    
    words = set(text.lower().split())
    marker_matches = words.intersection(IMPORTANT_MARKERS)
    
    # Base importance on marker density and position indicators
    score = len(marker_matches) * 0.2
    
    # Check for introductory or concluding phrases
    if any(text.lower().startswith(start) for start in ['first', 'next', 'then', 'finally', 'lastly']):
        score += 0.3
    
    return min(max(score, 0.0), 1.0)

@dataclass
class StorySegment:
    video_id: str
    start_time: float
    duration: float
    text: str
    energy_score: float
    is_key_moment: bool
    segment_type: str  # 'intro', 'discovery', 'reaction', 'transition', 'conclusion'

def analyze_story_structure(combined_transcript: List[Dict]) -> List[StorySegment]:
    """Analyze the combined transcript to identify key story moments"""
    
    story_segments = []
    
    # Story beat detection
    STORY_MARKERS = {
        'intro': ['what\'s up', 'hey', 'alright', 'introducing'],
        'discovery': ['look at this', 'whoa', 'found', 'check this out'],
        'reaction': ['wow', 'amazing', 'incredible', 'oh my'],
        'conclusion': ['thanks for', 'don\'t forget', 'see you next', 'that\'s it']
    }
    
    for video in combined_transcript:
        text = video['text'].lower()
        timestamp = video['start_time']
        
        # Determine segment type
        segment_type = 'transition'  # default
        for beat_type, markers in STORY_MARKERS.items():
            if any(marker in text for marker in markers):
                segment_type = beat_type
                break
        
        energy = calculate_energy_score(text)
        is_key = energy > 0.7 or segment_type in ['intro', 'discovery', 'conclusion']
        
        story_segments.append(StorySegment(
            video_id=video['video_id'],
            start_time=timestamp,
            duration=video['duration'],
            text=text,
            energy_score=energy,
            is_key_moment=is_key,
            segment_type=segment_type
        ))
    
    return story_segments

def generate_strategic_edit_points(story_segments: List[StorySegment]) -> List[Dict]:
    """Generate focused edit points for key story moments"""
    
    edit_points = []
    
    # Always include intro
    if story_segments:
        intro = next((seg for seg in story_segments if seg.segment_type == 'intro'), story_segments[0])
        edit_points.append({
            'timestamp': intro.start_time,
            'duration': min(3.0, intro.duration),
            'shot_type': 'wide_establishing',
            'transition': 'fade_in',
            'technical_notes': generate_ffmpeg_command('intro', intro.energy_score)
        })
    
    # Include key discoveries and reactions
    key_moments = [seg for seg in story_segments 
                  if seg.is_key_moment and seg.segment_type in ['discovery', 'reaction']]
    
    for moment in key_moments[:3]:  # Limit to top 3 key moments
        edit_points.append({
            'timestamp': moment.start_time,
            'duration': min(2.5, moment.duration),
            'shot_type': 'close_up' if moment.segment_type == 'discovery' else 'reaction',
            'transition': select_transition(moment.energy_score),
            'technical_notes': generate_ffmpeg_command(moment.segment_type, moment.energy_score)
        })
    
    # Always include conclusion
    conclusion = next((seg for seg in reversed(story_segments) 
                      if seg.segment_type == 'conclusion'), story_segments[-1])
    edit_points.append({
        'timestamp': conclusion.start_time,
        'duration': min(3.0, conclusion.duration),
        'shot_type': 'medium_wide',
        'transition': 'fade_out',
        'technical_notes': generate_ffmpeg_command('conclusion', conclusion.energy_score)
    })
    
    return edit_points

def generate_ffmpeg_command(segment_type: str, energy_score: float) -> str:
    """Generate FFmpeg command based on segment type and energy"""
    base_cmd = "ffmpeg -i input.mp4"
    
    effects = {
        'intro': "-vf 'fade=in:0:30,eq=brightness=0.1:saturation=1.2'",
        'discovery': "-vf 'eq=contrast=1.2:brightness=0.05'",
        'reaction': "-vf 'unsharp=3:3:1'",
        'conclusion': "-vf 'fade=out:0:30,eq=brightness=-0.05'"
    }
    
    return f"{base_cmd} {effects.get(segment_type, '')} output.mp4"

def process_from_edit_points(video_path: str, edit_points: List[Dict]) -> str:
    """Process video according to edit points"""
    try:
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Number of edit points: {len(edit_points)}")
        
        # Create absolute paths
        output_dir = os.path.abspath(app.config['OUTPUT_FOLDER'])
        temp_dir = os.path.abspath(os.path.join(output_dir, 'temp_segments'))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = os.path.join(output_dir, f"final_edited_video_{timestamp}.mp4")
        
        # Ensure temp directory exists
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process each segment
        segment_files = []
        for i, point in enumerate(edit_points):
            segment_output = os.path.join(temp_dir, f"segment_{i}.mp4")
            
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(point['timestamp']),
                '-t', str(point['duration']),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',
                '-async', '1',
                segment_output
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(segment_output):
                segment_files.append(segment_output)
                logger.info(f"Successfully processed segment {i}")
            else:
                logger.error(f"Error processing segment {i}: {result.stderr}")
        
        if not segment_files:
            logger.error("No segments were successfully processed")
            return None
            
        # Create concat file with absolute paths
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for file in segment_files:
                # Use absolute paths in concat file
                f.write(f"file '{os.path.abspath(file)}'\n")
        
        # Concatenate segments
        concat_command = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            final_output
        ]
        
        concat_result = subprocess.run(concat_command, capture_output=True, text=True)
        
        # Cleanup
        for file in segment_files + [concat_file]:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")
        
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Could not remove temp directory: {e}")
        
        if concat_result.returncode == 0 and os.path.exists(final_output):
            logger.info(f"Successfully created final video: {final_output}")
            return final_output
        else:
            logger.error(f"Error concatenating segments: {concat_result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def convert_to_wav(video_path: str) -> str:
    """Convert video file to WAV format for transcription"""
    try:
        # Create WAV path in same directory as video
        wav_path = os.path.splitext(video_path)[0] + '.wav'
        
        # FFmpeg command to extract audio as WAV
        command = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            wav_path
        ]
        
        # Run conversion
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            logger.info(f"Successfully converted {video_path} to WAV")
            return wav_path
        else:
            logger.error(f"Failed to convert to WAV: {result.stderr}")
            raise Exception("WAV conversion failed")
            
    except Exception as e:
        logger.error(f"Error converting to WAV: {str(e)}")
        raise

def process_video(files, style):
    try:
        # Create timestamped folder for this session
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(app.config['OUTPUT_FOLDER'], f'edited_video_{timestamp}')
        os.makedirs(session_folder, exist_ok=True)
        
        # Collect all transcriptions
        all_transcriptions = []
        video_paths = []
        total_duration = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.save(filepath)
                video_paths.append(filepath)
                
                # Get video duration and transcription
                probe = ffmpeg.probe(filepath)
                duration = float(probe['streams'][0]['duration'])
                
                wav_path = convert_to_wav(filepath)
                transcription = transcribe_audio(wav_path)
                
                all_transcriptions.append({
                    'text': transcription,
                    'start_time': total_duration,
                    'duration': duration,
                    'filename': filename
                })
                
                total_duration += duration
                os.remove(wav_path)
        
        # Create detailed analysis file
        analysis_path = os.path.join(session_folder, 'video_analysis.txt')
        with open(analysis_path, 'w') as f:
            f.write(f"Video Analysis Report\n{'='*20}\n\n")
            f.write(f"Style: {style}\n\n")
            
            f.write("Timeline:\n")
            for t in all_transcriptions:
                f.write(f"\nFile: {t['filename']}\n")
                f.write(f"Start Time: {t['start_time']:.2f}s\n")
                f.write(f"Duration: {t['duration']:.2f}s\n")
                f.write(f"Transcription:\n{t['text']}\n")
                f.write("-"*50 + "\n")
        
        # Generate more specific editing instructions
        edit_points = generate_edit_suggestions(all_transcriptions, style)
        
        # Add edit points to analysis file
        with open(analysis_path, 'a') as f:
            f.write("\nEdit Points:\n")
            for edit in edit_points:
                f.write(f"\nType: {edit['type']}\n")
                f.write(f"Timestamp: {edit['timestamp']:.2f}s\n")
                f.write(f"Duration: {edit['duration']:.2f}s\n")
                if 'speed_factor' in edit:
                    f.write(f"Speed Factor: {edit['speed_factor']}\n")
                f.write(f"Content: {edit['content']}\n")
                f.write("-"*30 + "\n")
        
        # Process videos with edit points
        final_output = os.path.join(session_folder, 'final_edited_video.mp4')
        
        success = process_videos_with_edit_points(video_paths, edit_points, session_folder)
        
        if success:
            return session_folder
        else:
            raise Exception("Video processing failed")
            
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        raise

def generate_edit_suggestions(transcriptions: List[Dict], style: str) -> List[Dict]:
    """Generate more specific and effective edit points"""
    try:
        client = OpenAI()
        
        # Debug log to see what we're receiving
        logger.info(f"Received transcriptions: {transcriptions}")
        
        # Create a timeline string for better context
        timeline = ""
        total_time = 0
        
        # Handle both string and dict transcriptions
        for t in transcriptions:
            if isinstance(t, dict):
                # If it's already a dictionary
                start_time = t.get('start_time', total_time)
                duration = t.get('duration', 0)
                text = t.get('text', '')
            else:
                # If it's a string
                text = str(t)
                start_time = total_time
                duration = 5  # Default duration if not specified
                
            timeline += f"Time {start_time:.1f}-{start_time + duration:.1f}: {text}\n"
            total_time += duration
        
        prompt = f"""As a professional video editor, create a sequence of precise editing instructions.

Style: {style}

Content Timeline:
{timeline}

Total Duration: {total_time:.1f} seconds

Create specific editing instructions in this exact format:
[
    {{
        "type": "cut",
        "timestamp": 0.0,
        "duration": 3.0,
        "content": "description"
    }},
    {{
        "type": "speed",
        "timestamp": 3.0,
        "duration": 2.0,
        "speed_factor": 1.5,
        "content": "description"
    }}
]

Rules:
1. All timestamps must be less than the total duration ({total_time:.1f} seconds)
2. Each segment must have a positive duration
3. Use only these edit types: "cut", "speed", "transition", "fade"
4. Include "speed_factor" only for "speed" type edits
5. Ensure smooth transitions between segments
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert video editor. Respond only with valid JSON arrays containing edit instructions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Parse the response
        try:
            content = response.choices[0].message.content.strip()
            # Remove any markdown code block indicators
            content = content.replace('```json', '').replace('```', '').strip()
            
            logger.info(f"Raw GPT response: {content}")
            
            edit_points = json.loads(content)
            
            if not isinstance(edit_points, list):
                logger.error("Response is not a list of edit points")
                return []
            
            # Validate each edit point
            validated_edits = []
            for edit in edit_points:
                if validate_edit_point(edit):
                    validated_edits.append(edit)
            
            logger.info(f"Validated edit points: {validated_edits}")
            return validated_edits
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {content}")
            return []
            
    except Exception as e:
        logger.error(f"Error generating edit suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def validate_edit_point(edit: Dict) -> bool:
    """Validate individual edit point"""
    try:
        # Required fields for all edit types
        required_fields = {'type', 'timestamp', 'duration', 'content'}
        if not all(field in edit for field in required_fields):
            logger.error(f"Missing required fields in edit point: {edit}")
            return False
        
        # Validate types
        if not isinstance(edit['timestamp'], (int, float)):
            logger.error(f"Invalid timestamp type: {type(edit['timestamp'])}")
            return False
        if not isinstance(edit['duration'], (int, float)):
            logger.error(f"Invalid duration type: {type(edit['duration'])}")
            return False
        if not isinstance(edit['content'], str):
            logger.error(f"Invalid content type: {type(edit['content'])}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating edit point: {str(e)}")
        return False

def generate_editing_instructions(transcripts: List[Dict]) -> List[Dict]:
    """
    Generate editing instructions based on the combined transcript of the uploaded footage.
    
    :param transcripts: List of dictionaries containing 'text', 'start_time', 'duration', and 'filename'.
    :return: List of editing instructions.
    """
    instructions = []
    current_time = 0

    for transcript in transcripts:
        text = transcript['text']
        start_time = transcript['start_time']
        duration = transcript['duration']
        filename = transcript['filename']

        # Example logic to determine key moments and transitions
        if "introduction" in text.lower():
            instructions.append({
                'type': 'cut',
                'timestamp': start_time,
                'duration': duration,
                'content': 'Introduction',
                'filename': filename
            })
        elif "transition" in text.lower():
            instructions.append({
                'type': 'transition',
                'timestamp': start_time,
                'duration': duration,
                'content': 'Transition',
                'filename': filename
            })
        elif "conclusion" in text.lower():
            instructions.append({
                'type': 'fade',
                'timestamp': start_time,
                'duration': duration,
                'content': 'Conclusion',
                'filename': filename
            })
        else:
            instructions.append({
                'type': 'cut',
                'timestamp': start_time,
                'duration': duration,
                'content': 'Scene',
                'filename': filename
            })

        current_time += duration

    return instructions

# Example usage
transcripts = [
    {'text': "Introduction to the rock hunt.", 'start_time': 0, 'duration': 10, 'filename': 'video1.mov'},
    {'text': "Transition to the basketball scene.", 'start_time': 10, 'duration': 5, 'filename': 'video2.mov'},
    {'text': "Conclusion of the video.", 'start_time': 15, 'duration': 5, 'filename': 'video3.mov'}
]

instructions = generate_editing_instructions(transcripts)
for instruction in instructions:
    print(instruction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))