from dataclasses import dataclass
from typing import List

@dataclass
class StyleGuide:
    name: str
    transition_duration: float
    cut_frequency: str
    effects: List[str]
    pacing: str

STYLE_GUIDES = {
    "Fast & Energized": StyleGuide(
        name="Fast & Energized",
        transition_duration=0.3,
        cut_frequency="2-4",
        effects=["fade=in:0:30", "fade=out:0:30"],
        pacing="rapid cuts with minimal transitions"
    ),
    "Moderate": StyleGuide(
        name="Moderate",
        transition_duration=0.75,
        cut_frequency="4-8",
        effects=["fade=in:0:45", "fade=out:0:45"],
        pacing="balanced cuts with smooth transitions"
    ),
    "Slow & Smooth": StyleGuide(
        name="Slow & Smooth",
        transition_duration=1.5,
        cut_frequency="8-15",
        effects=["fade=in:0:60", "fade=out:0:60"],
        pacing="gradual transitions with longer shots"
    )
}
