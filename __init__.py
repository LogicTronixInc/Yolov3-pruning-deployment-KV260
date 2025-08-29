import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))



from benchmark_model import *
from benchmark_utilities import *
from pruning_utils import *
from script_model import to_channels_last_safe
