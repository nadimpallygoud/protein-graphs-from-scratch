"""
Focused deep-dive Manim scenes for the repository.

Run with:
manim -pql utils/manim_animations_deep.py -a --format=gif
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.manim_storyboard import FeatureVectorUpdateScene as _FeatureVectorUpdateScene
from utils.manim_storyboard import LRPNumericConservationScene as _LRPNumericConservationScene
from utils.manim_storyboard import ProteinBackboneScene as _ProteinBackboneScene


class FeatureVectorUpdateScene(_FeatureVectorUpdateScene):
    pass


class LRPNumericConservationScene(_LRPNumericConservationScene):
    pass


class ProteinBackboneScene(_ProteinBackboneScene):
    pass

