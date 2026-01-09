"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      NONOGRAM TASK CONFIGURATION                              ║
║                                                                               ║
║  Configuration for nonogram (picross) task generation.                        ║
║  Inherits common settings from core.GenerationConfig                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from pydantic import Field
from core import GenerationConfig


class TaskConfig(GenerationConfig):
    """
    Nonogram task-specific configuration.
    
    Configuration for generating nonogram puzzles with various patterns and difficulty levels.
    
    Inherited from GenerationConfig:
        - num_samples: int          # Number of samples to generate
        - domain: str               # Task domain name (default: "nonogram")
        - difficulty: Optional[str] # Difficulty level (easy/medium/hard)
        - random_seed: Optional[int] # For reproducibility
        - output_dir: Path          # Where to save outputs
        - image_size: tuple[int, int] # Image dimensions
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    #  OVERRIDE DEFAULTS
    # ══════════════════════════════════════════════════════════════════════════
    
    domain: str = Field(default="nonogram")
    image_size: tuple[int, int] = Field(default=(768, 512))
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    generate_videos: bool = Field(
        default=True,
        description="Whether to generate ground truth videos"
    )
    
    video_fps: int = Field(
        default=10,
        description="Video frame rate"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    cell_size: int = Field(
        default=35,
        description="Size of each nonogram cell in pixels"
    )
    
    grid_line_width: int = Field(
        default=2,
        description="Width of grid lines"
    )
