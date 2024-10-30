# editing_instructions.py

# Define constants for edit types
CUT = 'cut'
SPEED = 'speed'
EFFECT = 'effect'
TRANSITION = 'transition'

# Define constants for effect names
ZOOM = 'zoom'
BLUR = 'blur'
HIGHLIGHT = 'highlight'

# Define constants for transition names
FADE = 'fade'
WIPE = 'wipe'

# Define intensity levels
LOW = 'low'
MEDIUM = 'medium'
HIGH = 'high'

def create_cut(start_time, end_time):
    return {
        'type': CUT,
        'start_time': start_time,
        'end_time': end_time
    }

def create_speed_change(start_time, end_time, rate):
    return {
        'type': SPEED,
        'start_time': start_time,
        'end_time': end_time,
        'parameters': {'rate': rate}
    }

def create_effect(start_time, end_time, name, intensity=MEDIUM):
    return {
        'type': EFFECT,
        'start_time': start_time,
        'end_time': end_time,
        'parameters': {'name': name, 'intensity': intensity}
    }

def create_transition(start_time, end_time, name, duration=0.5):
    return {
        'type': TRANSITION,
        'start_time': start_time,
        'end_time': end_time,
        'parameters': {'name': name, 'duration': duration}
    }

