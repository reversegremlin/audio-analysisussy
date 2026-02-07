"""Core audio processing modules."""

from chromascope.core.decomposer import AudioDecomposer
from chromascope.core.analyzer import FeatureAnalyzer
from chromascope.core.polisher import SignalPolisher

__all__ = ["AudioDecomposer", "FeatureAnalyzer", "SignalPolisher"]
