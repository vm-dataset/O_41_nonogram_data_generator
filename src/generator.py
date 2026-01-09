"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NONOGRAM TASK GENERATOR                                     ║
║                                                                               ║
║  Generates nonogram (picross/logic puzzle) tasks for video reasoning.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from core import BaseGenerator, TaskPair
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt

# Visual constants
GRID_LINE_COLOR = "#cbd5e1"
FILL_COLOR = "#1e293b"
EMPTY_COLOR = "white"
BACKGROUND_COLOR = "#f8fafc"
HINT_COLOR = "#64748b"


@dataclass
class NonogramPattern:
    """Specification for a nonogram pattern."""
    grid_size: int  # NxN grid
    pattern: np.ndarray  # 2D array: 0=empty, 1=filled
    row_hints: List[List[int]]  # Hints for each row
    col_hints: List[List[int]]  # Hints for each column
    pattern_type: str  # Type of pattern (cross, square, circle, etc.)


class TaskGenerator(BaseGenerator):
    """
    Nonogram task generator.
    
    Generates nonogram puzzles with different patterns and difficulty levels.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.cell_size = config.cell_size
        self.grid_line_width = config.grid_line_width
        self._seen_signatures: set[str] = set()
        
        # Initialize video generator if enabled
        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(fps=config.video_fps, output_format="mp4")
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate one nonogram task pair."""
        
        # Generate nonogram pattern
        difficulty = self.config.difficulty or self._choose_difficulty()
        grid_size = self._grid_size_for_difficulty(difficulty)
        pattern = self._create_pattern(grid_size, difficulty)
        
        # Ensure uniqueness
        signature = self._build_signature(pattern)
        max_attempts = 300
        attempts = 0
        original_size = grid_size
        
        while signature in self._seen_signatures and attempts < max_attempts:
            if attempts % 30 == 0 and attempts > 0:
                grid_size = self._grid_size_for_difficulty(difficulty)
            elif attempts % 10 == 0:
                grid_size = original_size
            
            pattern = self._create_pattern(grid_size, difficulty)
            signature = self._build_signature(pattern)
            attempts += 1
        
        if signature in self._seen_signatures:
            # If we can't generate unique, just continue (for now)
            pass
        else:
            self._seen_signatures.add(signature)
        
        # Render images
        first_image = self._render_start(pattern)
        final_image = self._render_end(pattern)
        
        # Generate video (optional)
        video_path = None
        if self.config.generate_videos and self.video_generator:
            video_path = self._generate_video(first_image, final_image, task_id, pattern)
        
        # Select prompt
        prompt = get_prompt(pattern.pattern_type)
        
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=video_path
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  RENDERING
    # ══════════════════════════════════════════════════════════════════════════
    
    def _render_start(self, pattern: NonogramPattern) -> Image.Image:
        """Render first frame: blank grid with hints."""
        return self._render_nonogram(pattern, show_solution=False)
    
    def _render_end(self, pattern: NonogramPattern) -> Image.Image:
        """Render final frame: complete solution."""
        return self._render_nonogram(pattern, show_solution=True)
    
    def _render_nonogram(self, pattern: NonogramPattern, show_solution: bool = False, filled_cells: Optional[Set[Tuple[int, int]]] = None) -> Image.Image:
        """
        Render nonogram grid with hints using matplotlib.
        
        Args:
            pattern: NonogramPattern to render
            show_solution: If True, show complete solution
            filled_cells: Optional set of (i, j) tuples for partially filled cells
        """
        w, h = self.config.image_size
        dpi = 150
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.invert_yaxis()
        
        # Draw background
        bg = Rectangle((0, 0), w, h, facecolor=BACKGROUND_COLOR, edgecolor="none")
        ax.add_patch(bg)
        
        size = pattern.grid_size
        cell_size = self.cell_size
        
        # Calculate hint area sizes
        max_row_hint_len = max(len(",".join(map(str, hints))) for hints in pattern.row_hints) if pattern.row_hints else 0
        max_col_hint_count = max(len(hints) for hints in pattern.col_hints) if pattern.col_hints else 0
        
        # Hint area dimensions
        row_hint_width = max(max_row_hint_len * 7 + 30, 50)
        col_hint_height = max(max_col_hint_count * 12 + 20, 40)
        
        # Position grid (centered, with hints outside)
        grid_start_x = row_hint_width + 20
        grid_start_y = col_hint_height + 20
        
        # Draw column hints (above grid)
        for j in range(size):
            x = grid_start_x + j * cell_size + cell_size / 2
            hints = pattern.col_hints[j]
            hint_text = "\n".join(map(str, hints)) if hints != [0] else "0"
            ax.text(x, grid_start_y - 10, hint_text, ha="center", va="bottom",
                    fontsize=9, color=HINT_COLOR, weight="bold", family="monospace")
        
        # Draw row hints (left of grid)
        for i in range(size):
            y = grid_start_y + i * cell_size + cell_size / 2
            hints = pattern.row_hints[i]
            hint_text = ",".join(map(str, hints)) if hints != [0] else "0"
            ax.text(grid_start_x - 10, y, hint_text, ha="right", va="center",
                    fontsize=9, color=HINT_COLOR, weight="bold", family="monospace")
        
        # Draw grid cells
        for i in range(size):
            for j in range(size):
                x = grid_start_x + j * cell_size
                y = grid_start_y + i * cell_size
                
                # Grid cell
                cell = Rectangle((x, y), cell_size, cell_size,
                               facecolor=EMPTY_COLOR, edgecolor=GRID_LINE_COLOR,
                               linewidth=self.grid_line_width)
                ax.add_patch(cell)
                
                # Fill if solution is shown and pattern indicates, or if in filled_cells
                should_fill = False
                if show_solution and pattern.pattern[i, j] == 1:
                    should_fill = True
                elif filled_cells is not None and (i, j) in filled_cells:
                    should_fill = True
                
                if should_fill:
                    fill = Rectangle((x + 2, y + 2), cell_size - 4, cell_size - 4,
                                   facecolor=FILL_COLOR, edgecolor="none")
                    ax.add_patch(fill)
        
        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, bbox_inches="tight", pad_inches=0.01, format='png')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        plt.close(fig)
        
        return img
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_video(
        self,
        first_image: Image.Image,
        final_image: Image.Image,
        task_id: str,
        pattern: NonogramPattern
    ) -> Optional[str]:
        """Generate ground truth video showing progressive cell filling."""
        import tempfile
        
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}_ground_truth.mp4"
        
        # Create animation frames showing progressive filling
        frames = self._create_nonogram_animation_frames(pattern)
        
        result = self.video_generator.create_video_from_frames(
            frames,
            video_path
        )
        
        return str(result) if result else None
    
    def _create_nonogram_animation_frames(
        self,
        pattern: NonogramPattern,
        hold_frames: int = 5,
        cells_per_frame: int = 2
    ) -> List[Image.Image]:
        """
        Create animation frames showing progressive cell filling.
        
        Fills cells progressively from top-left to bottom-right, revealing the pattern.
        """
        frames = []
        size = pattern.grid_size
        total_filled_cells = int(np.sum(pattern.pattern))
        
        # Get all filled cell positions
        filled_positions = []
        for i in range(size):
            for j in range(size):
                if pattern.pattern[i, j] == 1:
                    filled_positions.append((i, j))
        
        # Sort by row then column for natural filling order
        filled_positions.sort()
        
        # Hold initial empty state
        first_frame = self._render_start(pattern)
        for _ in range(hold_frames):
            frames.append(first_frame.copy())
        
        # Progressive filling frames
        # Fill cells in batches for smoother animation
        for batch_start in range(0, len(filled_positions), cells_per_frame):
            batch_end = min(batch_start + cells_per_frame, len(filled_positions))
            filled_cells = set(filled_positions[:batch_end])
            
            frame = self._render_nonogram(pattern, show_solution=False, filled_cells=filled_cells)
            frames.append(frame)
        
        # Ensure final frame is complete (show_solution=True gives exact final image)
        final_frame = self._render_end(pattern)
        frames.append(final_frame)
        
        # Hold final state
        for _ in range(hold_frames):
            frames.append(final_frame.copy())
        
        return frames
    
    # ══════════════════════════════════════════════════════════════════════════
    #  PATTERN GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _choose_difficulty(self) -> str:
        """Choose difficulty based on weights: 40% easy, 40% medium, 20% hard."""
        return random.choices(
            ["easy", "medium", "hard"],
            weights=[0.4, 0.4, 0.2]
        )[0]
    
    def _grid_size_for_difficulty(self, difficulty: str) -> int:
        """Determine grid size based on difficulty."""
        if difficulty == "easy":
            return random.choice([5, 6])
        if difficulty == "hard":
            return random.choice([12, 15])
        return random.choice([7, 8, 10])  # medium
    
    def _create_pattern(self, size: int, difficulty: str) -> NonogramPattern:
        """Generate a nonogram pattern."""
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            pattern_type = self._choose_pattern_type(size, difficulty)
            
            if pattern_type == "cross":
                pattern_array = self._generate_cross(size)
            elif pattern_type == "square":
                pattern_array = self._generate_square(size)
            elif pattern_type == "circle":
                pattern_array = self._generate_circle(size)
            elif pattern_type == "checkerboard":
                pattern_array = self._generate_checkerboard(size)
            elif pattern_type == "letter_t":
                pattern_array = self._generate_letter_t(size)
            elif pattern_type == "diagonal":
                pattern_array = self._generate_diagonal(size)
            else:  # random
                pattern_array = self._generate_random(size, difficulty)
            
            # Ensure no completely empty rows or columns
            if self._has_empty_row_or_column(pattern_array):
                attempts += 1
                continue
            
            row_hints, col_hints = self._calculate_hints(pattern_array)
            
            # Ensure no row or column has all zeros in hints
            if self._has_all_zero_hints(row_hints) or self._has_all_zero_hints(col_hints):
                attempts += 1
                continue
            
            return NonogramPattern(
                grid_size=size,
                pattern=pattern_array,
                row_hints=row_hints,
                col_hints=col_hints,
                pattern_type=pattern_type
            )
        
        # Fallback: ensure no empty rows/columns
        pattern_array = np.zeros((size, size), dtype=int)
        pattern_array = self._ensure_no_empty_rows_columns(pattern_array)
        row_hints, col_hints = self._calculate_hints(pattern_array)
        
        return NonogramPattern(
            grid_size=size,
            pattern=pattern_array,
            row_hints=row_hints,
            col_hints=col_hints,
            pattern_type="random"
        )
    
    def _choose_pattern_type(self, size: int, difficulty: str) -> str:
        """Choose pattern type based on grid size and difficulty."""
        if difficulty == "easy":
            if size <= 6:
                return random.choice(["cross", "square", "diagonal"])
            else:
                return random.choice(["cross", "square", "checkerboard"])
        elif difficulty == "hard":
            if size >= 12:
                return random.choice(["circle", "letter_t", "random"])
            else:
                return random.choice(["circle", "letter_t", "checkerboard", "random"])
        else:  # medium
            if size <= 8:
                return random.choice(["cross", "square", "circle", "checkerboard"])
            else:
                return random.choice(["circle", "letter_t", "checkerboard"])
    
    def _generate_cross(self, size: int) -> np.ndarray:
        """Generate a cross pattern."""
        pattern = np.zeros((size, size), dtype=int)
        center = size // 2
        
        # Add some randomness: sometimes offset the center slightly
        if random.random() < 0.3 and size > 5:
            offset = random.choice([-1, 1])
            center = max(1, min(size - 2, center + offset))
        
        # Vertical line
        pattern[:, center] = 1
        # Horizontal line
        pattern[center, :] = 1
        
        # Occasionally add small variations
        if random.random() < 0.1 and size >= 7:
            corner = random.choice([(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)])
            pattern[corner[0], corner[1]] = 1
        
        return pattern
    
    def _generate_square(self, size: int) -> np.ndarray:
        """Generate a hollow square pattern."""
        pattern = np.zeros((size, size), dtype=int)
        # Border only
        pattern[0, :] = 1  # Top
        pattern[-1, :] = 1  # Bottom
        pattern[:, 0] = 1  # Left
        pattern[:, -1] = 1  # Right
        
        # Occasionally add a diagonal
        if random.random() < 0.15 and size >= 7:
            if random.random() < 0.5:
                # Add main diagonal
                for i in range(size):
                    pattern[i, i] = 1
            else:
                # Add anti-diagonal
                for i in range(size):
                    pattern[i, size - 1 - i] = 1
        
        return pattern
    
    def _generate_circle(self, size: int) -> np.ndarray:
        """Generate a circular pattern (approximated)."""
        pattern = np.zeros((size, size), dtype=int)
        center = size / 2
        
        # Vary radius slightly for uniqueness
        base_radius = min(size, size) / 2 - 1
        radius_variation = random.uniform(-0.3, 0.3)
        radius = base_radius + radius_variation
        
        # Vary thickness for uniqueness
        thickness = random.uniform(0.6, 1.0)
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if abs(dist - radius) < thickness:
                    pattern[i, j] = 1
        return pattern
    
    def _generate_checkerboard(self, size: int) -> np.ndarray:
        """Generate a checkerboard pattern."""
        pattern = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    pattern[i, j] = 1
        return pattern
    
    def _generate_letter_t(self, size: int) -> np.ndarray:
        """Generate a letter T pattern."""
        pattern = np.zeros((size, size), dtype=int)
        # Top horizontal bar
        bar_width = max(3, size // 3)
        bar_width += random.choice([-1, 0, 1])
        bar_width = max(2, min(size - 2, bar_width))
        start_col = (size - bar_width) // 2
        pattern[0, start_col:start_col + bar_width] = 1
        
        # Vertical stem
        stem_col = size // 2
        if random.random() < 0.3 and size > 6:
            stem_col += random.choice([-1, 1])
            stem_col = max(1, min(size - 2, stem_col))
        pattern[:, stem_col] = 1
        return pattern
    
    def _generate_diagonal(self, size: int) -> np.ndarray:
        """Generate a diagonal line pattern."""
        pattern = np.zeros((size, size), dtype=int)
        
        diagonal_type = random.choice(["main", "anti", "both"])
        
        if diagonal_type == "main" or diagonal_type == "both":
            for i in range(size):
                pattern[i, i] = 1
        if diagonal_type == "anti" or diagonal_type == "both":
            for i in range(size):
                pattern[i, size - 1 - i] = 1
        
        return pattern
    
    def _generate_random(self, size: int, difficulty: str) -> np.ndarray:
        """Generate a random pattern with controlled density."""
        pattern = np.zeros((size, size), dtype=int)
        
        # Density based on difficulty
        if difficulty == "easy":
            density = random.uniform(0.25, 0.35)
        elif difficulty == "hard":
            density = random.uniform(0.55, 0.70)
        else:
            density = random.uniform(0.40, 0.50)
        
        # For easy difficulty, prefer patterns with fewer, longer blocks
        if difficulty == "easy":
            num_blocks = random.randint(2, max(3, size // 2))
            total_filled = 0
            target_filled = int(size * size * density)
            
            for _ in range(num_blocks):
                if total_filled >= target_filled * 0.9:
                    break
                # Create a horizontal or vertical block
                if random.random() < 0.5:
                    # Horizontal block
                    row = random.randint(0, size - 1)
                    col_start = random.randint(0, max(1, size - 3))
                    block_length = random.randint(2, min(4, size - col_start))
                    pattern[row, col_start:col_start + block_length] = 1
                    total_filled += block_length
                else:
                    # Vertical block
                    col = random.randint(0, size - 1)
                    row_start = random.randint(0, max(1, size - 3))
                    block_length = random.randint(2, min(4, size - row_start))
                    pattern[row_start:row_start + block_length, col] = 1
                    total_filled += block_length
        else:
            # For medium/hard: more random distribution
            num_filled = int(size * size * density)
            positions = random.sample(range(size * size), num_filled)
            for pos in positions:
                i = pos // size
                j = pos % size
                pattern[i, j] = 1
        
        return pattern
    
    # ══════════════════════════════════════════════════════════════════════════
    #  HELPER METHODS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _has_empty_row_or_column(self, pattern: np.ndarray) -> bool:
        """Check if pattern has any completely empty rows or columns."""
        size = pattern.shape[0]
        # Check rows
        for i in range(size):
            if np.sum(pattern[i, :]) == 0:
                return True
        # Check columns
        for j in range(size):
            if np.sum(pattern[:, j]) == 0:
                return True
        return False
    
    def _has_all_zero_hints(self, hints: List[List[int]]) -> bool:
        """Check if all hints are [0] (completely empty)."""
        return all(h == [0] for h in hints)
    
    def _ensure_no_empty_rows_columns(self, pattern: np.ndarray) -> np.ndarray:
        """Ensure no row or column is completely empty by adding at least one cell."""
        size = pattern.shape[0]
        # Check and fix rows
        for i in range(size):
            if np.sum(pattern[i, :]) == 0:
                j = random.randint(0, size - 1)
                pattern[i, j] = 1
        # Check and fix columns
        for j in range(size):
            if np.sum(pattern[:, j]) == 0:
                i = random.randint(0, size - 1)
                pattern[i, j] = 1
        return pattern
    
    def _calculate_hints(self, pattern: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
        """Calculate row and column hints from pattern."""
        size = pattern.shape[0]
        
        # Row hints
        row_hints = []
        for i in range(size):
            row = pattern[i, :]
            hints = []
            count = 0
            for j in range(size):
                if row[j] == 1:
                    count += 1
                else:
                    if count > 0:
                        hints.append(count)
                        count = 0
            if count > 0:
                hints.append(count)
            row_hints.append(hints if hints else [0])
        
        # Column hints
        col_hints = []
        for j in range(size):
            col = pattern[:, j]
            hints = []
            count = 0
            for i in range(size):
                if col[i] == 1:
                    count += 1
                else:
                    if count > 0:
                        hints.append(count)
                        count = 0
            if count > 0:
                hints.append(count)
            col_hints.append(hints if hints else [0])
        
        return row_hints, col_hints
    
    def _build_signature(self, pattern: NonogramPattern) -> str:
        """Build a unique signature for the pattern."""
        pattern_str = ''.join(str(int(x)) for x in pattern.pattern.flatten())
        return f"{pattern.pattern_type}_{pattern.grid_size}_{pattern_str}"
