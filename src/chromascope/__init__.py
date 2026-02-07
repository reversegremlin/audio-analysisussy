"""Audio analysis engine for reactive generative art."""

from chromascope.core.decomposer import AudioDecomposer
from chromascope.core.analyzer import FeatureAnalyzer
from chromascope.core.polisher import SignalPolisher
from chromascope.io.exporter import ManifestExporter
from chromascope.pipeline import AudioPipeline

__version__ = "0.1.0"
__all__ = [
    "AudioDecomposer",
    "FeatureAnalyzer",
    "SignalPolisher",
    "ManifestExporter",
    "AudioPipeline",
]
