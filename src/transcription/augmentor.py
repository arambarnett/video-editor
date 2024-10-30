from typing import List
import logging
from .models.transcript import TranscriptSegment, EditInstruction, AugmentedTranscript
from .models.style_guide import STYLE_GUIDES, StyleGuide

logger = logging.getLogger(__name__)

class TranscriptAugmentor:
    def __init__(self, style: str = "Moderate"):
        self.style_guide = STYLE_GUIDES.get(style, STYLE_GUIDES["Moderate"])
    
    def augment(self, segments: List[TranscriptSegment]) -> AugmentedTranscript:
        try:
            edit_instructions = []
            current_time = 0
            
            for segment in segments:
                instruction = self._generate_instruction(
                    segment,
                    current_time,
                    self.style_guide
                )
                edit_instructions.append(instruction)
                current_time += segment.duration
            
            return AugmentedTranscript(
                segments=segments,
                edit_instructions=edit_instructions,
                style=self.style_guide.name,
                total_duration=current_time
            )
            
        except Exception as e:
            logger.error(f"Error augmenting transcript: {str(e)}")
            raise
    
    def _generate_instruction(
        self,
        segment: TranscriptSegment,
        current_time: float,
        style_guide: StyleGuide
    ) -> EditInstruction:
        return EditInstruction(
            type="cut",
            timestamp=current_time,
            duration=segment.duration,
            effect=style_guide.effects[0] if style_guide.effects else None
        )
