import pytest
from pathlib import Path
import os
from src.pipeline.base import PipelineContext
from src.pipeline.stages.input import VideoInputStage

@pytest.fixture
def sample_video(tmp_path):
    # Create a dummy video file for testing
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"dummy video content")
    return str(video_path)

@pytest.mark.asyncio
async def test_video_input_stage(sample_video):
    # Create context with test video
    context = PipelineContext(
        input_files=[sample_video],
        output_dir="output"
    )
    
    # Initialize and run input stage
    input_stage = VideoInputStage()
    result = await input_stage.process(context)
    
    # Verify results
    assert len(result.input_files) == 1
    assert result.input_files[0] == sample_video
    assert result.metadata is not None
