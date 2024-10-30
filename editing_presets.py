from moviepy.editor import vfx, CompositeAudioClip
import numpy as np
import cv2

# Custom transition functions
def custom_fade(clip, duration):
    return clip.fx(vfx.fadeout, duration=duration).fx(vfx.fadein, duration=duration)

def custom_dissolve(clip, duration):
    return clip.fx(vfx.fadeout, duration=duration).fx(vfx.fadein, duration=duration)

def custom_slide(clip, duration):
    def slide_func(get_frame, t):
        frame = get_frame(t)
        return frame[:, int(t/duration * frame.shape[1]):, :]
    return clip.fl(slide_func, apply_to=['mask'])

# Custom camera movement functions
def custom_speedx(clip, factor):
    return clip.speedx(factor)

def custom_crop(clip, x1, x2, y1, y2):
    w, h = clip.size
    return clip.crop(x1=int(w*x1), x2=int(w*x2), y1=int(h*y1), y2=int(h*y2))

def custom_rotate(clip, angle):
    return clip.rotate(angle)

def custom_zoom(clip, factor):
    def zoom(get_frame, t):
        img = get_frame(t)
        center = (img.shape[0]//2, img.shape[1]//2)
        M = cv2.getRotationMatrix2D(center, 0, factor)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return clip.fl(zoom)

def safe_speedx(clip, factor):
    return clip.speedx(factor) if clip.duration > 1/factor else clip

def safe_crop(clip, x1, x2, y1, y2):
    return clip.crop(x1=x1, x2=x2, y1=y1, y2=y2) if clip.w > 100 and clip.h > 100 else clip

def safe_rotate(clip, angle):
    return clip.rotate(angle) if clip.duration > 0.1 else clip

def safe_apply_effect(clip, effect_func):
    try:
        return effect_func(clip)
    except Exception:
        return clip

PRESETS = {
    "Fast & Energized": {
        "cut_style": {
            "average_duration": (2, 3),
            "transitions": {
                lambda clip: safe_apply_effect(clip, lambda c: custom_fade(c, 0.5)): 0.5,
                lambda clip: clip: 0.3,
                lambda clip: safe_apply_effect(clip, lambda c: c.fx(vfx.slide_out, duration=0.5)): 0.2
            }
        },
        "camera_movements": {
            lambda clip: safe_apply_effect(clip, lambda c: safe_speedx(c, 1.2)): 0.5,
            lambda clip: safe_apply_effect(clip, lambda c: safe_crop(c, 0.1, 0.9, 0, 1)): 0.3,
            lambda clip: safe_apply_effect(clip, lambda c: safe_rotate(c, 10)): 0.2
        },
        "audio_editing": {
            "audio_ducking": lambda voiceover, video_duration: voiceover.set_duration(video_duration)
        }
    },
    "Moderate": {
        "cut_style": {
            "average_duration": (5, 7),
            "transitions": {
                lambda clip: custom_fade(clip, 1): 0.4,
                lambda clip: custom_dissolve(clip, 1): 0.3,
                lambda clip: custom_slide(clip, 1): 0.3
            }
        },
        "camera_movements": {
            lambda clip: clip.fx(vfx.pan, x1=0.1, x2=0.9): 0.4,
            lambda clip: clip.fx(vfx.rotate, angle=10): 0.3,
            lambda clip: clip.fx(vfx.zoom, factor=1.1): 0.3
        },
        "audio_editing": {
            "audio_ducking": lambda voiceover, video_duration: voiceover.set_duration(video_duration)
        }
    },
    "Slow & Smooth": {
        "cut_style": {
            "average_duration": (10, 15),
            "transitions": {
                lambda clip: custom_fade(clip, 2): 0.5,
                lambda clip: custom_dissolve(clip, 2): 0.3,
                lambda clip: custom_slide(clip, 2): 0.2
            }
        },
        "camera_movements": {
            lambda clip: clip.fx(vfx.zoom, factor=1.1): 0.5,
            lambda clip: clip.fx(vfx.pan, x1=0.1, x2=0.9): 0.3,
            lambda clip: clip.fx(vfx.rotate, angle=10): 0.2
        },
        "audio_editing": {
            "audio_ducking": lambda voiceover, video_duration: voiceover.set_duration(video_duration)
        }
    }
}
