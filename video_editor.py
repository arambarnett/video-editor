import os
import subprocess
from moviepy.editor import *
from editing_presets import PRESETS
import random
class VideoEditor:
    def __init__(self, method='moviepy'):
        self.method = method

    def edit_video(self, input_file, output_file, style):
        if self.method == 'moviepy':
            return self.edit_with_moviepy(input_file, style)
        elif self.method == 'ffmpeg':
            return self.edit_with_ffmpeg(input_file, output_file, style)
        elif self.method == 'kdenlive':
            return self.edit_with_kdenlive(input_file, output_file, style)
        else:
            raise ValueError(f"Unsupported editing method: {self.method}")

    def edit_with_moviepy(self, input_file, output_file, style):
        try:
            preset = PRESETS[style]
            video = VideoFileClip(input_file)
            
            # Validate video loaded correctly
            if not video.duration:
                raise ValueError("Invalid video file")

            # Apply edits with error checking
            try:
                # Apply cuts and transitions
                cut_duration = min(random.uniform(*preset["cut_style"]["average_duration"]), video.duration)
                transition = random.choices(list(preset["cut_style"]["transitions"].keys()),
                                         weights=list(preset["cut_style"]["transitions"].values()))[0]
                video = video.subclip(0, cut_duration).fx(transition)
                
                # Apply camera movement
                camera_movement = random.choices(list(preset["camera_movements"].keys()),
                                                 weights=list(preset["camera_movements"].values()))[0]
                video = camera_movement(video)
                
                # Apply audio editing
                background_music = preset["audio_editing"]["music"](video.duration)
                sound_effect = preset["audio_editing"]["sound_effects"]()
                final_audio = preset["audio_editing"]["audio_ducking"](background_music, video.audio, video.duration)
                video = video.set_audio(final_audio)
                
                video.write_videofile(output_file)
                return output_file
            except Exception as e:
                raise ValueError(f"Error applying edits: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading video or preset: {str(e)}")

    def edit_with_ffmpeg(self, input_file, output_file, style):
        # This is a placeholder. You'll need to implement FFmpeg commands based on the style
        cmd = f"ffmpeg -i {input_file} -vf 'scale=1280:720' {output_file}"
        subprocess.run(cmd, shell=True, check=True)
        return output_file

    def edit_with_kdenlive(self, input_file, output_file, style):
        # This is a placeholder. You'll need to implement Kdenlive CLI commands or API calls
        print(f"Kdenlive editing not implemented yet. Style: {style}")
        return input_file  # Return input file as no editing is done

