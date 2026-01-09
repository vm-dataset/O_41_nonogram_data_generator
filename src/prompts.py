"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        NONOGRAM TASK PROMPTS                                  ║
║                                                                               ║
║  Prompts/instructions for nonogram (picross) puzzle tasks.                    ║
║  Prompts are selected based on pattern type and returned to the model.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random


# ══════════════════════════════════════════════════════════════════════════════
#  DEFINE YOUR PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "default": [
        "Solve this nonogram puzzle by filling in the grid cells according to the row and column hints. The numbers on the left indicate the lengths of consecutive filled blocks in each row, and the numbers on top indicate the lengths of consecutive filled blocks in each column. Fill in the cells to reveal the hidden pattern. Keep the camera view fixed in the top-down perspective and maintain the grid structure unchanged. Stop the video when all cells are correctly filled and the complete pattern is revealed.",
    ],
}


def get_prompt(task_type: str = "default") -> str:
    """
    Select a random prompt for the given task type.
    
    Args:
        task_type: Type of task (key in PROMPTS dict)
        
    Returns:
        Random prompt string from the specified type
    """
    prompts = PROMPTS.get(task_type, PROMPTS["default"])
    return random.choice(prompts)


def get_all_prompts(task_type: str = "default") -> list[str]:
    """Get all prompts for a given task type."""
    return PROMPTS.get(task_type, PROMPTS["default"])
