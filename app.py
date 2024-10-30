from flask import Flask, request, jsonify, send_file, url_for, send_from_directory, render_template
from flask_cors import CORS
import os
import logging
import uuid
from datetime import datetime
import subprocess
import zipfile
from openai import OpenAI
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, TypedDict
import json
import ffmpeg
from google.cloud import storage
from google.oauth2 import service_account
import psutil
import shlex
import shutil
import difflib
import glob
import openai
from pathlib import Path
import time
import random
import string


# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app without size restrictions
app = Flask(__name__)
CORS(app)

# Update CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Add OPTIONS handler for preflight requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Basic configuration
app.config.update(
    UPLOAD_FOLDER=os.path.abspath('uploads'),
    OUTPUT_FOLDER=os.path.abspath('output'),
    STATIC_FOLDER=os.path.abspath('static'),
    TEMPLATE_FOLDER=os.path.abspath('templates')
)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Google Cloud Storage client
try:
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    )
    storage_client = storage.Client(credentials=credentials)
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    bucket = storage_client.bucket(bucket_name)
    logger.info("Successfully initialized Google Cloud Storage client")
except Exception as e:
    logger.error(f"Error initializing Google Cloud Storage: {str(e)}")
    storage_client = None
    bucket = None

def upload_to_gcs(file_path: str, destination_blob_name: str) -> str:
    """Upload a file to Google Cloud Storage"""
    try:
        if not storage_client or not bucket:
            logger.error("Google Cloud Storage not initialized")
            return None

        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        
        # Make the blob publicly readable
        blob.make_public()
        
        return blob.public_url
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        return None

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'wmv', 'flv', 'mkv', 'm4v', 'webm'}

STYLE_GUIDES = {
    "Fast & Energized": {
        "transition_duration": 0.3,
        "cut_frequency": "2-4",
        "effects": ["fade=in:0:30", "fade=out:0:30", "crossfade=duration=0.3"],
        "pacing": "rapid cuts with minimal transitions"
    },
    "Moderate": {
        "transition_duration": 0.75,
        "cut_frequency": "4-8",
        "effects": ["fade=in:0:45", "fade=out:0:45", "crossfade=duration=0.75"],
        "pacing": "balanced cuts with smooth transitions"
    },
    "Slow & Smooth": {
        "transition_duration": 1.5,
        "cut_frequency": "8-15",
        "effects": ["fade=in:0:60", "fade=out:0:60", "crossfade=duration=1.5"],
        "pacing": "gradual transitions with longer shots"
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    if exclude is None:
        exclude = []
    
    try:
        for filepath in file_paths:
            if filepath and os.path.exists(filepath) and filepath not in exclude:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Type definitions
VideoData = Dict[str, Any]
TranscriptionSegment = Dict[str, Any]
EditPoint = Dict[str, Any]

class StyleGuide(TypedDict):
    transition_duration: float
    cut_frequency: float
    pacing: str

def extract_metadata(video_path: str) -> dict:
    """Extract duration and other metadata from video file"""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(probe['format']['duration'])
        
        metadata = {
            'duration': duration,
            'width': int(video_info['width']),
            'height': int(video_info['height']),
            'fps': eval(video_info['r_frame_rate'])
        }
        
        logger.info(f"Successfully extracted metadata for {video_path}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        raise

def extract_audio(video_path: str) -> str:
    """Extract audio from video file"""
    try:
        audio_path = video_path.rsplit('.', 1)[0] + '.mp3'
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ab', '192k',
            '-ar', '44100',
            '-ac', '2',
            audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Successfully extracted audio from {video_path}")
        return audio_path
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with open(audio_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            
            # Extract text from response
            transcription = str(response)
            logger.info(f"Successfully transcribed {audio_path}")
            return transcription

    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise

def analyze_content(video_data: List[Dict]) -> Dict:
    """Analyze video content to determine natural edit points"""
    try:
        full_text = ""
        segments = []
        total_duration = 0
        
        # Collect all transcriptions and segments
        for video in video_data:
            transcription = video['transcription']
            full_text += transcription['text'] + " "
            
            # Get base duration for this video
            duration = float(video['metadata']['format']['duration'])
            total_duration += duration
            
            # Split into meaningful segments based on natural breaks
            text_segments = transcription['text'].split('.')
            segment_duration = duration / len(text_segments)
            
            start_time = 0
            for text in text_segments:
                if text.strip():  # Skip empty segments
                    segments.append({
                        'video_index': len(segments),
                        'text': text.strip(),
                        'start_time': start_time,
                        'duration': segment_duration,
                        'energy_level': analyze_segment_energy(text.strip())
                    })
                start_time += segment_duration

        # Generate key points
        key_points = [
            f"Total Duration: {total_duration:.1f} seconds",
            f"Number of Segments: {len(segments)}",
            f"Key Moments: {identify_key_moments(segments)}"
        ]

        return {
                'full_text': full_text.strip(),
            'segments': segments,
            'key_points': key_points,
            'total_duration': total_duration,
            'video_count': len(video_data)
        }

    except Exception as e:
        logger.error(f"Error in analyze_content: {str(e)}")
        raise

def analyze_segment_energy(text: str) -> float:
    """Analyze text content to determine energy level (0.0 to 1.0)"""
    # High energy indicators
    high_energy_patterns = {
        'exclamation': ['!', 'wow', 'amazing', 'awesome', 'incredible'],
        'action': ['found', 'look', 'check', 'watch', 'see', 'going', 'let\'s'],
        'discovery': ['found', 'discovered', 'spotted', 'there it is', 'here it is'],
        'excitement': ['beautiful', 'perfect', 'rare', 'special', 'finally'],
        'emphasis': ['very', 'really', 'super', 'extremely', 'definitely']
    }
    
    # Low energy indicators
    low_energy_patterns = {
        'contemplative': ['thinking', 'maybe', 'might', 'could', 'perhaps'],
        'searching': ['looking', 'searching', 'trying to find'],
        'uncertainty': ['not sure', 'possibly', 'probably'],
        'transition': ['now', 'next', 'moving', 'going to']
    }
    
    text = text.lower()
    energy_score = 0.5  # Start neutral
    
    # Calculate energy based on content
    for category, patterns in high_energy_patterns.items():
        for pattern in patterns:
            if pattern in text:
                energy_score += 0.1
                logger.info(f"High energy pattern found: {pattern}")
    
    for category, patterns in low_energy_patterns.items():
        for pattern in patterns:
            if pattern in text:
                energy_score -= 0.1
                logger.info(f"Low energy pattern found: {pattern}")
    
    # Normalize between 0 and 1
    return max(0.0, min(1.0, energy_score))

def identify_key_moments(segments: List[Dict]) -> str:
    """Identify and describe key moments in the video"""
    key_moments = []
    for segment in segments:
        if segment['energy_level'] > 0.7:  # High energy moments
            key_moments.append(f"Highlight at {segment['start_time']:.1f}s")
    return ', '.join(key_moments) if key_moments else "Consistent pacing throughout"

def apply_style_effects(video_data: List[Dict], style: str) -> List[Dict]:
    """Generate dynamic edit points based on content analysis and style"""
    try:
        if not video_data:
            logger.error("No video data provided")
            return []

        logger.info(f"Applying {style} style to {len(video_data)} videos")
        edit_points = []

        # Base style modifiers
        style_modifiers = {
            "Fast & Energized": {
                'speed_range': (1.0, 1.5),      # Faster range
                'saturation_boost': 0.3,        # More vibrant
                'contrast_boost': 0.2,          # More punchy
                'energy_threshold': 0.6         # More likely to speed up
            },
            "Moderate": {
                'speed_range': (0.9, 1.2),      # Balanced range
                'saturation_boost': 0.1,        # Slight enhancement
                'contrast_boost': 0.1,          # Slight enhancement
                'energy_threshold': 0.5         # Neutral threshold
            },
            "Slow & Smooth": {
                'speed_range': (0.7, 1.0),      # Slower range
                'saturation_boost': 0.0,        # Natural colors
                'contrast_boost': 0.0,          # Natural contrast
                'energy_threshold': 0.4         # More likely to slow down
            }
        }

        modifiers = style_modifiers.get(style, style_modifiers["Moderate"])

        # Process each video
        for video_index, video in enumerate(video_data):
            try:
                # Analyze content energy
                text = video['transcription']['text']
                energy_level = analyze_segment_energy(text)
                duration = float(video['metadata']['format']['duration'])
                
                logger.info(f"Video {video_index} energy level: {energy_level}")

                # Dynamic speed based on content energy and style
                speed_range = modifiers['speed_range']
                if energy_level > modifiers['energy_threshold']:
                    # Higher energy = faster
                    speed = speed_range[1]
                else:
                    # Lower energy = slower
                    speed = speed_range[0]

                # Dynamic visual effects based on content
                saturation = 1.0 + (energy_level * modifiers['saturation_boost'])
                contrast = 1.0 + (energy_level * modifiers['contrast_boost'])

                edit_point = {
                    'file_index': video_index,
                    'start_time': 0,
                    'duration': duration,
                    'energy_level': energy_level,
                    'effects': {
                        'speed': speed,
                        'saturation': saturation,
                        'contrast': contrast
                    }
                }
                
                edit_points.append(edit_point)
                logger.info(f"Generated dynamic edit point for video {video_index + 1}")
                logger.info(f"Effects: speed={speed:.2f}, sat={saturation:.2f}, con={contrast:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing video {video_index}: {e}")
                continue

        return edit_points

    except Exception as e:
        logger.error(f"Error in apply_style_effects: {str(e)}")
        raise

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"

def write_analysis_report(
    analysis_path: str, 
    video_data: List[Dict], 
    analysis: Dict, 
    edit_points: List[Dict],
    session_id: str,
    timestamp: str,
    style: str
) -> None:
    """Write detailed analysis report with summary and edit points"""
    try:
        with open(analysis_path, 'w') as f:
            f.write("Video Analysis Report\n")
            f.write("===================\n\n")
            
            # Session Info
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Style: {style}\n")
            f.write(f"Files processed: {', '.join([os.path.basename(v['path']) for v in video_data])}\n\n")
            
            # Content Summary
            f.write("Content Summary\n")
            f.write("--------------\n")
            full_transcript = " ".join([segment['text'] for video in video_data 
                                     for segment in video['transcription']['segments']])
            f.write(f"Full Transcript: {full_transcript}\n\n")
            
            # Key Points
            f.write("Key Points\n")
            f.write("----------\n")
            for point in analysis.get('key_points', []):
                f.write(f"- {point}\n")
            f.write("\n")
            
            # Edit Points
            f.write("Edit Points:\n")
            f.write(json.dumps(edit_points, indent=2))
            f.write("\n")

    except Exception as e:
        logger.error(f"Error writing analysis report: {str(e)}")
        raise

def convert_to_wav(video_path: str) -> str:
    """Convert video audio to WAV format for transcription"""
    try:
        wav_path = f"{video_path}_audio.wav"
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, wav_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True)
        return wav_path
    except Exception as e:
        logger.error(f"Error converting to WAV: {str(e)}")
        raise

def process_single_video(file, session_folder: str) -> Dict:
    """Process a single video file and return its information"""
    try:
        # Save original filename
        original_filename = secure_filename(file.filename)
        
        # Save the uploaded file
        video_path = os.path.join(session_folder, original_filename)
        file.save(video_path)
        
        logger.info(f"Processing video file: {video_path}")

        # Get video metadata using FFprobe
        metadata = get_video_metadata(video_path)
        
        # Get transcription
        transcription = transcribe_audio(video_path)
        
        return {
            'name': original_filename,  # Store original filename
            'path': video_path,
            'metadata': metadata,
            'transcription': transcription
        }

    except Exception as e:
        logger.error(f"Error processing video {file.filename}: {e}")
        raise

def check_for_duplicates(video_data: List[Dict]) -> List[Dict]:
    """Remove duplicate video segments based on content similarity"""
    unique_videos = []
    seen_content = set()
    
    for video in video_data:
        # Get text content and create a simplified version for comparison
        content = video['transcription']['text'].lower().strip()
        # Remove common filler words for better comparison
        simplified = ' '.join(word for word in content.split() 
                            if word not in {'um', 'uh', 'like', 'so', 'and', 'but'})
        
        # Check if we've seen very similar content before
        is_duplicate = False
        for seen in seen_content:
            # Using difflib to check similarity ratio
            if difflib.SequenceMatcher(None, simplified, seen).ratio() > 0.8:
                is_duplicate = True
                logger.info(f"Detected duplicate content: {content[:50]}...")
                break
        
        if not is_duplicate:
            seen_content.add(simplified)
            unique_videos.append(video)
            
    logger.info(f"Removed {len(video_data) - len(unique_videos)} duplicate segments")
    return unique_videos

class VideoEditor:
    def __init__(self, session_folder: str):
        self.session_folder = session_folder
        self.logger = logging.getLogger(__name__)

    def create_edit_script(self, edit_points: List[Dict], video_data: List[Dict]) -> str:
        """Generate FFmpeg filter complex script from edit points"""
        script_parts = []
        outputs = []
        
        for idx, edit in enumerate(edit_points):
            video = video_data[edit['file_index']]
            
            # Input label
            input_label = f"[{idx}:v]"
            audio_label = f"[{idx}:a]"
            
            # Video filters
            video_filters = []
            
            # Trim if needed
            if 'start_time' in edit and 'duration' in edit:
                video_filters.append(f"trim=start={edit['start_time']}:duration={edit['duration']}")
                video_filters.append("setpts=PTS-STARTPTS")  # Reset timestamps
            
            # Speed adjustment
            if edit['effects'].get('speed') and edit['effects']['speed'] != 1.0:
                speed = edit['effects']['speed']
                video_filters.append(f"setpts={1/speed}*PTS")
            
            # Visual effects
            if edit['effects'].get('saturation') or edit['effects'].get('contrast'):
                eq_params = []
                if edit['effects'].get('saturation'):
                    eq_params.append(f"saturation={edit['effects']['saturation']}")
                if edit['effects'].get('contrast'):
                    eq_params.append(f"contrast={edit['effects']['contrast']}")
                video_filters.append(f"eq={':'.join(eq_params)}")
            
            # Audio filters
            audio_filters = []
            
            # Trim audio to match video
            if 'start_time' in edit and 'duration' in edit:
                audio_filters.append(f"atrim=start={edit['start_time']}:duration={edit['duration']}")
                audio_filters.append("asetpts=PTS-STARTPTS")
            
            # Speed adjustment for audio
            if edit['effects'].get('speed') and edit['effects']['speed'] != 1.0:
                audio_filters.append(f"atempo={edit['effects']['speed']}")
            
            # Combine filters
            if video_filters:
                script_parts.append(f"{input_label}{','.join(video_filters)}[v{idx}]")
            else:
                script_parts.append(f"{input_label}null[v{idx}]")
                
            if audio_filters:
                script_parts.append(f"{audio_label}{','.join(audio_filters)}[a{idx}]")
            else:
                script_parts.append(f"{audio_label}anull[a{idx}]")
            
            outputs.extend([f"[v{idx}]", f"[a{idx}]"])
        
        # Concatenate all segments
        video_concat = ''.join(f"[v{i}]" for i in range(len(edit_points)))
        audio_concat = ''.join(f"[a{i}]" for i in range(len(edit_points)))
        
        script_parts.append(f"{video_concat}concat=n={len(edit_points)}:v=1:a=0[vout]")
        script_parts.append(f"{audio_concat}concat=n={len(edit_points)}:v=0:a=1[aout]")
        
        return ';'.join(script_parts)

    def apply_edits(self, edit_points: List[Dict], video_data: List[Dict], output_path: str) -> None:
        """Apply edits using FFmpeg"""
        try:
            # Build FFmpeg command
            cmd = ['ffmpeg', '-y']
            
            # Add input files
            for video in video_data:
                cmd.extend(['-i', video['path']])
            
            # Generate filter complex script
            filter_script = self.create_edit_script(edit_points, video_data)
            self.logger.info(f"Generated FFmpeg filter script: {filter_script}")
            
            # Add filter complex and output options
            cmd.extend([
                '-filter_complex', filter_script,
                '-map', '[vout]',
                '-map', '[aout]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                output_path
            ])
            
            # Execute FFmpeg command
            self.logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("FFmpeg failed to create output file")
                
            self.logger.info(f"Successfully created edited video: {output_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg processing failed: {e.stderr}")
            raise RuntimeError(f"Failed to process video: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Error applying edits: {str(e)}")
            raise

def process_video(files, style: str) -> Dict:
    try:
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(app.config['OUTPUT_FOLDER'], f'session_{session_id}_{timestamp}')
        os.makedirs(session_folder, exist_ok=True)

        logger.info(f"Starting video processing with style: {style}")
        
        # 1. Process videos and get metadata + transcriptions
        video_data = []
        for file in files:
            video_info = process_single_video(file, session_folder)
            video_data.append(video_info)
            logger.info(f"Processed video: {file.filename}")

        # 2. Remove duplicates
        video_data = check_for_duplicates(video_data)
        logger.info(f"Processing {len(video_data)} unique video segments")

        # 3. Generate edit points based on content and style
        edit_points = apply_style_effects(video_data, style)
        logger.info(f"Generated {len(edit_points)} edit points")

        # 4. Process each segment with effects
        processed_segments = []
        for idx, edit_point in enumerate(edit_points):
            video = video_data[edit_point['file_index']]
            segment_path = os.path.join(session_folder, f'segment_{idx}.mp4')
            
            logger.info(f"Applying effects to segment {idx + 1}/{len(edit_points)}")
            
            # Extract segment if needed
            if 'start_time' in edit_point and 'duration' in edit_point:
                extract_cmd = [
                    'ffmpeg', '-y',
                    '-i', video['path'],
                    '-ss', str(edit_point['start_time']),
                    '-t', str(edit_point['duration']),
                    '-c', 'copy',
                    segment_path + '.temp'
                ]
                subprocess.run(extract_cmd, check=True, capture_output=True)
                input_path = segment_path + '.temp'
            else:
                input_path = video['path']

            # Apply effects
            apply_ffmpeg_effects(input_path, segment_path, edit_point['effects'])
            processed_segments.append(segment_path)
            logger.info(f"Processed segment {idx + 1}")

        # 5. Concatenate all processed segments
        logger.info("Concatenating processed segments...")
        final_output = os.path.join(session_folder, f'final_edited_video_{session_id}.mp4')
        
        # Create concat file
        concat_file = os.path.join(session_folder, 'concat.txt')
        with open(concat_file, 'w') as f:
            for segment in processed_segments:
                f.write(f"file '{segment}'\n")

        # Concatenate using FFmpeg
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            final_output
        ]
        subprocess.run(concat_cmd, check=True, capture_output=True)
        logger.info("Created final video")

        # 6. Save session data for download
        session_data = {
            'session_id': session_id,
            'style': style,
            'video_data': video_data,
            'edit_points': edit_points
        }
        
        with open(os.path.join(session_folder, 'session_data.json'), 'w') as f:
            json.dump(session_data, f, indent=2)

        # 7. Create analysis report
        report = create_analysis_report(session_id, video_data, edit_points, style)
        with open(os.path.join(session_folder, 'video_analysis_report.txt'), 'w') as f:
            f.write(report)

        # 8. Clean up temporary files
        for temp_file in glob.glob(os.path.join(session_folder, '*.temp')):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")

        logger.info("Video processing complete")
        return {
            'success': True,
            'session_id': session_id,
            'download_url': f'/api/download/{session_id}',
            'analysis': {
                'full_text': ' '.join(v['transcription']['text'] for v in video_data),
                'video_count': len(video_data),
                'key_points': [
                    f"Processed {len(video_data)} unique segments",
                    f"Applied {style} style effects",
                    f"Total edit points: {len(edit_points)}"
                ]
            }
        }

    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        raise

def handle_error(e: Exception, context: str) -> Dict:
    """Centralized error handling"""
    error_message = f"Error in {context}: {str(e)}"
    logger.error(error_message)
    return {
        "error": error_message,
        "context": context,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.route('/api/process', methods=['POST'])
def process_videos():
    try:
        # Log the incoming request
        app.logger.info(f"Received request: {request.files}")
        app.logger.info(f"Form data: {request.form}")

        # Create session ID and timestamp
        session_id = ''.join(random.choices(string.hexdigits.lower(), k=8))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_folder = f'session_{session_id}_{timestamp}'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], session_folder)
        os.makedirs(output_path, exist_ok=True)

        # Get style from form data
        style = request.form.get('style', 'Fast & Energized')

        # Process each video file
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400

        # ... rest of your video processing code ...

        return jsonify({
            'message': 'Processing complete',
            'session_id': session_id,
            'timestamp': timestamp
        })

    except Exception as e:
        app.logger.error(f"Error processing videos: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>/<timestamp>')
def download_file(session_id, timestamp):
    """Handle file downloads"""
    try:
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], f'session_{session_id}_{timestamp}.zip')
        if os.path.exists(zip_path):
            return send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'edited_video_{session_id}_{timestamp}.zip'
            )
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error in download endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Serve React App
@app.route('/')
def serve_frontend():
    return send_from_directory(app.config['STATIC_FOLDER'], 'index.html')

# Serve static files
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.config['STATIC_FOLDER'], path)

# Move API routes under /api prefix
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/info')
def api_info():
    return jsonify({
        'status': 'online',
        'endpoints': {
            'process': '/api/process',
            'download': '/download/<session_id>/<timestamp>'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'available_styles': list(STYLE_GUIDES.keys())
    })

# Add error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found on this server.',
        'endpoints': {
            'process': '/api/process',
            'download': '/download/<session_id>/<timestamp>'
        }
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(e)
    }), 500

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'error': 'Method Not Allowed',
        'message': 'The method is not allowed for the requested URL.',
        'allowed_methods': e.valid_methods
    }), 405

# Admin dashboard routes
@app.route('/admin')
def admin_dashboard():
    return render_template('admin/dashboard.html')

@app.route('/admin/stats')
def get_admin_stats():
    try:
        now = datetime.now()
        stats = {
            'total_videos': 100,  # Replace with actual count
            'processing_time': '2.5s',  # Replace with actual average
            'last_updated': now.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'operational'
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting admin stats: {str(e)}")
        raise

@app.route('/admin/test', methods=['POST'])
def run_load_test():
    try:
        file_count = int(request.form.get('fileCount', 1))
        file_size = int(request.form.get('fileSize', 1))
        concurrent = int(request.form.get('concurrent', 1))
        
        # Implement your load testing logic here
        # This is a placeholder response
        return jsonify({
            'avgTime': 2.5,
            'successRate': 95,
            'memoryUsage': 256
        })
    except Exception as e:
        logger.error(f"Error running load test: {str(e)}")
        return jsonify({'error': str(e)}), 500

def extract_json_from_response(response_content: str, expected_start: str = '[') -> str:
    """Extract valid JSON from OpenAI response"""
    try:
        # Clean the response
        content = response_content.strip()
        
        # If it's already valid JSON, return it
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
            
        # Try to find JSON in the response
        start_idx = content.find(expected_start)
        if start_idx == -1:
            raise ValueError(f"No {expected_start} found in response")
            
        # Find matching end character
        end_char = ']' if expected_start == '[' else '}'
        end_idx = content.rfind(end_char) + 1
        
        if end_idx <= 0:
            raise ValueError(f"No matching {end_char} found in response")
            
        json_content = content[start_idx:end_idx]
        
        # Validate extracted content
        json.loads(json_content)  # This will raise JSONDecodeError if invalid
        return json_content
        
    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}")
        logger.error(f"Original content: {response_content}")
        raise

# Configure logging to show more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verify FFmpeg installation
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    logger.info("FFmpeg and FFprobe are properly installed")
except subprocess.CalledProcessError as e:
    logger.error("FFmpeg or FFprobe is not properly installed")
    raise RuntimeError("FFmpeg or FFprobe is not properly installed")

def create_analysis_report(session_id: str, video_data: List[Dict], edit_points: List[Dict], style: str) -> str:
    """Create a comprehensive analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = f"""Video Analysis Report
===================

Session ID: {session_id}
Timestamp: {timestamp}
Style: {style}
Files processed: {', '.join(v['name'] for v in video_data)}

Content Summary
--------------
Full Transcript: {' '.join(v['transcription']['text'] for v in video_data)}

Key Points
----------"""

    # Add key points from analysis
    analysis = analyze_content(video_data)
    for point in analysis['key_points']:
        report += f"\n- {point}"

    report += "\n\nDetailed Transcriptions\n---------------------"
    
    # Add detailed transcriptions with timestamps
    for video in video_data:
        report += f"\nFile: {video['name']}\n"
        report += "Timestamps | Content\n-----------|---------\n"
        for segment in video['transcription']['segments']:
            start = float(segment.get('start', 0))
            end = float(segment.get('end', 0))
            report += f"{format_timestamp(start)} - {format_timestamp(end)} | \"{segment['text']}\"\n"

    report += "\n\nEdit Decisions\n-------------"
    
    # Add edit decisions
    for idx, edit in enumerate(edit_points):
        video = video_data[edit['file_index']]
        start = format_timestamp(edit['start_time'])
        end = format_timestamp(edit['start_time'] + edit['duration'])
        report += f"\nTimestamp: {start} - {end}\n"
        report += f"Content: \"{video['transcription']['text']}\"\n"
        report += f"Edit Type: speed={edit['effects']['speed']:.1f}, "
        report += f"saturation={edit['effects']['saturation']:.1f}, "
        report += f"contrast={edit['effects']['contrast']:.1f}\n"

    report += "\n\nFFmpeg Edit Instructions\n----------------------\n"
    report += json.dumps(edit_points, indent=2)

    return report

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

@app.route('/api/download/<session_id>', methods=['GET'])
def download_results(session_id):
    try:
        # Find the session folder
        session_pattern = f'session_{session_id}_*'
        session_folders = glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], session_pattern))
        
        if not session_folders:
            raise FileNotFoundError(f"No session found for ID: {session_id}")
            
        session_folder = session_folders[0]
        logger.info(f"Found session folder: {session_folder}")

        # Create a temporary folder for the download package
        output_folder = os.path.join(session_folder, 'download_package')
        os.makedirs(output_folder, exist_ok=True)

        # Load session data
        with open(os.path.join(session_folder, 'session_data.json'), 'r') as f:
            session_data = json.load(f)

        # Copy final video
        final_video = session_data['final_video']
        if os.path.exists(final_video):
            shutil.copy2(final_video, os.path.join(output_folder, 'final_edited_video.mp4'))
            logger.info("Copied final video to download package")

        # Copy analysis report
        report_path = os.path.join(session_folder, 'video_analysis_report.txt')
        if os.path.exists(report_path):
            shutil.copy2(report_path, output_folder)
            logger.info("Copied analysis report to download package")

        # Create zip file
        shutil.make_archive(output_folder, 'zip', output_folder)
        
        return send_file(
            f'{output_folder}.zip',
            as_attachment=True,
            download_name=f'edited_video_package_{session_id}.zip',
            mimetype='application/zip'
        )

    except Exception as e:
        logger.error(f"Error creating download package: {str(e)}")
        return jsonify({'error': str(e)}), 500

def concat_files(input_files: List[str], output_file: str) -> None:
    """Concatenate video files using FFmpeg"""
    try:
        # Create concat file
        concat_file = output_file + '.txt'
        with open(concat_file, 'w') as f:
            for file in input_files:
                f.write(f"file '{file}'\n")

        # Concatenate videos
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            output_file
        ]
        
        logger.info(f"Running concat command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Clean up concat file
        os.remove(concat_file)
        
    except Exception as e:
        logger.error(f"Error in concat_files: {str(e)}")
        raise

def get_video_metadata(video_path: str) -> Dict:
    """Get video metadata using FFprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        logger.info(f"Successfully extracted metadata for {video_path}")
        return metadata
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFprobe failed: {e.stderr}")
        raise RuntimeError(f"Failed to get video metadata: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse FFprobe output: {e}")
        raise RuntimeError("Invalid FFprobe output format")
    except Exception as e:
        logger.error(f"Error in get_video_metadata: {str(e)}")
        raise

def transcribe_video(video_path: str) -> Dict:
    """Transcribe video audio using OpenAI Whisper API"""
    try:
        # Extract audio from video
        audio_path = video_path + '.wav'
        extract_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            audio_path
        ]
        
        subprocess.run(extract_cmd, capture_output=True, check=True)
        logger.info(f"Extracted audio from {video_path}")
        
        # Transcribe using OpenAI API
        with open(audio_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up audio file
        os.remove(audio_path)
        
        logger.info(f"Successfully transcribed {video_path}")
        
        # Format the response
        transcription_data = {
            'text': response.text,
            'segments': []  # Initialize empty segments if not provided
        }
        
        # If response has segments, process them
        if hasattr(response, 'segments'):
            transcription_data['segments'] = [
                {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                }
                for segment in response.segments
            ]
        
        return transcription_data
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
        raise RuntimeError(f"Failed to extract audio: {e.stderr}")
    except Exception as e:
        logger.error(f"Error in transcribe_video: {str(e)}")
        raise

def analyze_content_for_edits(video_data: List[Dict], style: str) -> List[Dict]:
    """Use GPT to analyze transcripts and generate intelligent edit points"""
    try:
        # Prepare the content analysis prompt
        segments = []
        for idx, video in enumerate(video_data):
            segments.append({
                'index': idx,
                'text': video['transcription']['text'],  # Use simple text
                'duration': float(video['metadata']['format']['duration'])
            })

        prompt = f"""As a video editing expert, analyze these video segments and suggest edits.
Style guide: {style}

Video segments:
{json.dumps(segments, indent=2)}

For each segment, provide:
1. Suggested speed (0.5-2.0x) based on content
2. Visual effects (saturation and contrast adjustments)
3. Reason for the edit decisions
4. Timestamp ranges to focus on

Respond in this JSON format:
{{
    "edits": [
        {{
            "segment_index": 0,
            "speed": 1.2,
            "effects": {{"saturation": 1.1, "contrast": 1.1}},
            "reasoning": "High energy moment needs emphasis",
            "timestamp_range": [0, 10.5]
        }}
    ]
}}"""

        # Get editing suggestions from GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional video editor."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )

        # Parse the response
        analysis = json.loads(response.choices[0].message.content)
        logger.info(f"Generated edit suggestions: {json.dumps(analysis, indent=2)}")

        # Convert GPT suggestions to edit points
        edit_points = []
        for edit in analysis['edits']:
            edit_point = {
                'file_index': edit['segment_index'],
                'start_time': edit['timestamp_range'][0],
                'duration': edit['timestamp_range'][1] - edit['timestamp_range'][0],
                'effects': {
                    'speed': edit['speed'],
                    'saturation': edit['effects']['saturation'],
                    'contrast': edit['effects']['contrast']
                },
                'reasoning': edit['reasoning']
            }
            edit_points.append(edit_point)
            logger.info(f"Added edit point: {json.dumps(edit_point, indent=2)}")

        return edit_points

    except Exception as e:
        logger.error(f"Error in analyze_content_for_edits: {str(e)}")
        raise

def apply_ffmpeg_effects(input_path: str, output_path: str, effects: Dict, start_time: float = 0, duration: float = None) -> None:
    """Apply FFmpeg effects to video segment"""
    try:
        # Build filter chains for video and audio separately
        video_filters = []
        audio_filters = []
        
        # Speed effect
        if effects.get('speed') and effects['speed'] != 1.0:
            video_filters.append(f"setpts={1/effects['speed']}*PTS")
            audio_filters.append(f"atempo={effects['speed']}")
        
        # Visual effects
        if effects.get('saturation') or effects.get('contrast'):
            eq_params = []
            if effects.get('saturation'):
                eq_params.append(f"saturation={effects['saturation']}")
            if effects.get('contrast'):
                eq_params.append(f"contrast={effects['contrast']}")
            video_filters.append(f"eq={':'.join(eq_params)}")

        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Add trim if specified
        if duration:
            cmd.extend(['-ss', str(start_time), '-t', str(duration)])
            
        # Input file
        cmd.extend(['-i', input_path])
        
        # Add filter complex if we have any filters
        filter_complex = []
        if video_filters:
            filter_complex.append(f"[0:v]{','.join(video_filters)}[v]")
        if audio_filters:
            filter_complex.append(f"[0:a]{','.join(audio_filters)}[a]")
            
        if filter_complex:
            cmd.extend(['-filter_complex', ';'.join(filter_complex)])
            # Map the filtered streams
            if video_filters:
                cmd.extend(['-map', '[v]'])
            if audio_filters:
                cmd.extend(['-map', '[a]'])
        else:
            # If no filters, map the streams directly
            cmd.extend(['-map', '0:v', '-map', '0:a'])
        
        # Output settings
        cmd.extend([
            '-c:v', 'libx264',     # Video codec
            '-preset', 'medium',    # Encoding speed preset
            '-crf', '23',          # Quality (lower = better, 23 is default)
            '-c:a', 'aac',         # Audio codec
            '-b:a', '192k',        # Audio bitrate
            '-movflags', '+faststart',  # Web playback optimization
            output_path
        ])
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Execute FFmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Verify output file was created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("FFmpeg failed to create output file")
            
        logger.info(f"Successfully applied effects to {input_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg processing failed: {e.stderr}")
        raise RuntimeError(f"Failed to process video: {e.stderr}")
    except Exception as e:
        logger.error(f"Error in apply_ffmpeg_effects: {str(e)}")
        raise

@app.route('/api/debug_session/<session_id>')
def debug_session(session_id):
    # Look for any directory matching this session ID pattern
    base_path = "/app/output"
    session_dirs = [d for d in os.listdir(base_path) if d.startswith(f"session_{session_id}_")]
    
    if not session_dirs:
        return jsonify({"error": f"No session directory found for {session_id}"})
        
    session_dir = os.path.join(base_path, session_dirs[0])
    session_data_path = os.path.join(session_dir, "session_data.json")
    
    response = {
        "session_exists": True,
        "data_exists": os.path.exists(session_data_path),
        "session_dir": session_dir,
        "files": os.listdir(session_dir)
    }
    return jsonify(response)

def get_session_path(session_id):
    """Find the most recent session directory for a given ID"""
    base_path = Path("/app/output")
    session_pattern = f"session_{session_id}_*"
    
    try:
        # List all matching session directories
        matching_sessions = list(base_path.glob(session_pattern))
        if not matching_sessions:
            logger.error(f"No sessions found for ID: {session_id}")
            return None
            
        # Sort by creation time (most recent first)
        latest_session = max(matching_sessions, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found session directory: {latest_session}")
        return str(latest_session)
    except Exception as e:
        logger.error(f"Error finding session: {str(e)}")
        return None

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    try:
        session_dir = get_session_path(session_id)
        if not session_dir:
            logger.error(f"Session directory not found for ID: {session_id}")
            return jsonify({"error": f"No session found for ID: {session_id}"}), 404
            
        session_data_path = Path(session_dir) / "session_data.json"
        if not session_data_path.exists():
            logger.error(f"Session data file not found: {session_data_path}")
            return jsonify({"error": "Session data not found"}), 404
            
        with open(session_data_path) as f:
            session_data = json.load(f)
            
        logger.info(f"Successfully retrieved session data for ID: {session_id}")
        return jsonify(session_data)
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def generate_edit_points(full_text: str, video_data: List[dict], style: str) -> List[dict]:
    """Generate edit points based on transcript and style"""
    try:
        # Prepare prompt for OpenAI
        prompt = f"""
        Given this video transcript: "{full_text}"
        Style requested: {style}
        
        Create edit points that:
        1. Maintain narrative flow
        2. Match the {style} style
        3. Include appropriate effects for each segment
        
        Format each edit point as:
        {{
            "file_index": [video index],
            "start_time": [start in seconds],
            "duration": [duration in seconds],
            "effects": {{
                "speed": [0.5-2.0],
                "saturation": [0.8-1.5],
                "contrast": [0.8-1.5]
            }},
            "reasoning": [why this edit works]
        }}
        """

        # Get edit suggestions from OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional video editor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # Parse response into edit points
        edit_points = json.loads(response.choices[0].message.content)
        logger.info(f"Generated {len(edit_points)} edit points")
        
        return edit_points

    except Exception as e:
        logger.error(f"Error generating edit points: {str(e)}")
        raise

def process_segment(input_path: str, start_time: float, duration: float, effects: dict, output_path: str) -> str:
    """Process video segment with specified effects"""
    try:
        speed = effects.get('speed', 1.0)
        saturation = effects.get('saturation', 1.0)
        contrast = effects.get('contrast', 1.0)
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', input_path,
            '-filter_complex',
            f'[0:v]setpts={1/speed}*PTS,eq=saturation={saturation}:contrast={contrast}[v];[0:a]atempo={speed}[a]',
            '-map', '[v]',
            '-map', '[a]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]
        
        # Run FFmpeg with full error output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Verify output file
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Failed to create segment: {output_path}")
            
        logger.info(f"Successfully processed segment to {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise

@app.route('/output/session_<session_id>_<timestamp>/final_edited_video_<video_id>.mp4')
def serve_video(session_id, timestamp, video_id):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], 
                             f'session_{session_id}_{timestamp}', 
                             f'final_edited_video_{video_id}.mp4')
    return send_file(video_path, mimetype='video/mp4')

@app.route('/download/<session_id>/<timestamp>')
def download_video(session_id, timestamp):
    video_path = os.path.join(
        app.config['OUTPUT_FOLDER'],
        f'session_{session_id}_{timestamp}',
        f'final_edited_video_{session_id}.mp4'
    )
    if os.path.exists(video_path):
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=f'edited_video_{session_id}.mp4'
        )
    else:
        return "Video not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)