"""
Full Manim teaching sequence for the GNN protein explainability pipeline.

Run with:
manim -pql utils/manim_animations_full.py -a --format=gif
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.manim_storyboard import FeatureVectorUpdateScene as _FeatureVectorUpdateScene
from utils.manim_storyboard import GCNMessagePassingScene as _GCNMessagePassingScene
from utils.manim_storyboard import GNNLRPRelevantWalkScene as _GNNLRPRelevantWalkScene
from utils.manim_storyboard import GraphBasicsScene as _GraphBasicsScene
from utils.manim_storyboard import GraphPoolingScene as _GraphPoolingScene
from utils.manim_storyboard import LRPNumericConservationScene as _LRPNumericConservationScene
from utils.manim_storyboard import ProteinBackboneScene as _ProteinBackboneScene
from utils.manim_storyboard import ProteinDistanceGraphScene as _ProteinDistanceGraphScene
from utils.manim_storyboard import PyGEdgeIndexScene as _PyGEdgeIndexScene
from utils.manim_storyboard import SaliencyVsSubgraphScene as _SaliencyVsSubgraphScene
from utils.manim_storyboard import WildtypeMutantDeltaScene as _WildtypeMutantDeltaScene


class GraphBasicsScene(_GraphBasicsScene):
    pass


class GCNMessagePassingScene(_GCNMessagePassingScene):
    pass


class FeatureVectorUpdateScene(_FeatureVectorUpdateScene):
    pass


class PyGEdgeIndexScene(_PyGEdgeIndexScene):
    pass


class ProteinBackboneScene(_ProteinBackboneScene):
    pass


class ProteinDistanceGraphScene(_ProteinDistanceGraphScene):
    pass


class GraphPoolingScene(_GraphPoolingScene):
    pass


class SaliencyVsSubgraphScene(_SaliencyVsSubgraphScene):
    pass


class LRPNumericConservationScene(_LRPNumericConservationScene):
    pass


class GNNLRPRelevantWalkScene(_GNNLRPRelevantWalkScene):
    pass


class WildtypeMutantDeltaScene(_WildtypeMutantDeltaScene):
    pass

