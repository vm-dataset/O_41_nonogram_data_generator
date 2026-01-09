# Nonogram Task Data Generator 🧩

A Python tool for generating synthetic nonogram (picross/logic puzzle) tasks for video reasoning datasets. This generator creates nonogram puzzles with various patterns, difficulty levels, and automatically generates solution videos.

This task generator follows the [template-data-generator](https://github.com/vm-dataset/template-data-generator.git) format and is compatible with [VMEvalKit](https://github.com/Video-Reason/VMEvalKit.git).

Repository: [O_41_nonogram_data_generator](https://github.com/vm-dataset/O_41_nonogram_data_generator)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vm-dataset/O_41_nonogram_data_generator.git
cd O_41_nonogram_data_generator

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Generate nonogram tasks
python examples/generate.py --num-samples 50
```

---

## 📁 Project Structure

```
nonogram-task-data-generator/
├── core/                    # Core utilities
│   ├── base_generator.py   # Abstract base class
│   ├── schemas.py          # Pydantic models
│   ├── image_utils.py      # Image helpers
│   ├── video_utils.py      # Video generation
│   └── output_writer.py    # File output
├── src/                     # Nonogram-specific logic
│   ├── generator.py        # Nonogram task generator
│   ├── prompts.py          # Prompt templates
│   └── config.py           # Configuration
├── examples/
│   └── generate.py         # Entry point
└── data/questions/         # Generated output
```

---

## 📦 Output Format

Each generated task produces:

```
data/questions/nonogram_task/{task_id}/
├── first_frame.png          # Initial state with hints (REQUIRED)
├── final_frame.png          # Complete solution (REQUIRED)
├── prompt.txt               # Instructions (REQUIRED)
└── ground_truth.mp4         # Solution video (OPTIONAL)
```

---

## 🎨 Features

### Pattern Types
- **Cross**: Simple cross patterns
- **Square**: Hollow square borders
- **Circle**: Circular patterns (approximated)
- **Checkerboard**: Alternating pattern
- **Letter T**: Letter T shapes
- **Diagonal**: Diagonal lines
- **Random**: Random patterns with controlled density

### Difficulty Levels
- **Easy**: 5x5 to 6x6 grids, simple patterns
- **Medium**: 7x7 to 10x10 grids, moderate complexity
- **Hard**: 12x12 to 15x15 grids, complex patterns

### Automatic Hint Generation
- Row hints (left side): Show consecutive block lengths for each row
- Column hints (top): Show consecutive block lengths for each column
- Hints are automatically calculated from the pattern

### Video Generation
- Progressive cell filling animation
- Shows step-by-step solution process
- Configurable frame rate and animation speed

---

## ⚙️ Configuration

All configuration is in `src/config.py`:

```python
class TaskConfig(GenerationConfig):
    domain: str = "nonogram"
    image_size: tuple[int, int] = (768, 512)
    
    # Video settings
    generate_videos: bool = True
    video_fps: int = 10
    
    # Nonogram-specific settings
    cell_size: int = 35              # Size of each cell in pixels
    grid_line_width: int = 2         # Grid line thickness
```

---

## 📝 Usage Examples

### Basic Generation
```bash
# Generate 50 nonogram tasks
python examples/generate.py --num-samples 50
```

### Custom Output Directory
```bash
# Generate 100 tasks to custom directory
python examples/generate.py --num-samples 100 --output data/my_nonograms
```

### With Seed for Reproducibility
```bash
# Generate with fixed seed
python examples/generate.py --num-samples 50 --seed 42
```

### Without Video Generation
```bash
# Generate tasks without solution videos
python examples/generate.py --num-samples 50 --no-videos
```

---

## 🎯 How Nonograms Work

A nonogram (also known as picross or griddlers) is a logic puzzle where:

1. **Grid**: You have a rectangular grid (typically square)
2. **Hints**: Each row and column has numbers indicating the lengths of consecutive filled blocks
3. **Goal**: Fill in the cells to reveal a hidden pattern
4. **Rules**: 
   - Numbers indicate groups of consecutive filled cells
   - Groups are separated by at least one empty cell
   - Order matters: hints are read left-to-right for rows, top-to-bottom for columns

### Example
For a row with hint `[3, 2]`:
- There are two groups of filled cells
- First group: 3 consecutive filled cells
- Second group: 2 consecutive filled cells
- At least one empty cell between groups

---

## 🔧 Customization

### Adding New Pattern Types

Edit `src/generator.py` and add a new pattern generation method:

```python
def _generate_my_pattern(self, size: int) -> np.ndarray:
    """Generate your custom pattern."""
    pattern = np.zeros((size, size), dtype=int)
    # Your pattern logic here
    return pattern
```

Then add it to `_create_pattern()` method.

### Customizing Prompts

Edit `src/prompts.py` to add or modify prompts:

```python
PROMPTS = {
    "default": [
        "Solve this nonogram puzzle...",
        "Fill in the grid according to the hints...",
    ],
    "cross": [
        "Reveal the cross pattern...",
    ],
}
```

---

## 📋 Requirements

- Python >= 3.8
- numpy
- Pillow
- pydantic
- matplotlib
- opencv-python (for video generation)

See `requirements.txt` for exact versions.

---

## 📄 License

See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
