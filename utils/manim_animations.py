"""
Compatibility entrypoint for the original three-scene teaching sequence.

Run with:
manim -pql utils/manim_animations.py <SceneName>
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.manim_storyboard import GCNMessagePassingScene as _GCNMessagePassingScene
from utils.manim_storyboard import GNNLRPRelevantWalkScene as _GNNLRPRelevantWalkScene
from utils.manim_storyboard import ProteinDistanceGraphScene as _ProteinDistanceGraphScene


class ProteinGraphConstructionScene(_ProteinDistanceGraphScene):
    pass


class MessagePassingScene(_GCNMessagePassingScene):
    pass


class GNNLRPScene(_GNNLRPRelevantWalkScene):
    pass

