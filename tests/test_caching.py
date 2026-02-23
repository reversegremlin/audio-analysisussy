"""Tests for AudioPipeline caching mechanism."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from chromascope.pipeline import AudioPipeline


def test_pipeline_caching(temp_audio_file, tmp_path, monkeypatch):
    """Test that caching works correctly (miss, hit, bypass)."""
    
    # Redirect cache dir to tmp_path so we don't touch real user cache
    monkeypatch.setattr(AudioPipeline, "_get_cache_dir", lambda self: tmp_path)
    
    pipeline = AudioPipeline(target_fps=60)
    
    # 1. First run: Should analyze and save to cache
    # We use wraps=pipeline.decompose to execute the real method but track calls
    with patch.object(pipeline, 'decompose', wraps=pipeline.decompose) as mock_decompose:
        result1 = pipeline.process(temp_audio_file)
        assert mock_decompose.called
        assert "manifest" in result1
    
    # Verify cache file exists
    cache_files = list(tmp_path.glob("manifest_*.json"))
    assert len(cache_files) == 1
    
    # 2. Second run: Should load from cache (decompose NOT called)
    with patch.object(pipeline, 'decompose', wraps=pipeline.decompose) as mock_decompose:
        result2 = pipeline.process(temp_audio_file)
        assert not mock_decompose.called, "Should have used cache"
        assert result2["manifest"] == result1["manifest"]
        
    # 3. Run with use_cache=False: Should re-analyze
    with patch.object(pipeline, 'decompose', wraps=pipeline.decompose) as mock_decompose:
        result3 = pipeline.process(temp_audio_file, use_cache=False)
        assert mock_decompose.called, "Should have re-analyzed"
        # The result should be effectively the same (floating point diffs aside)
        assert result3["bpm"] == result1["bpm"]


def test_cache_differentiation(temp_audio_file, tmp_path, monkeypatch):
    """Test that different FPS/SampleRates/Config create different cache files."""
    monkeypatch.setattr(AudioPipeline, "_get_cache_dir", lambda self: tmp_path)
    
    # Run with FPS 60
    pipeline60 = AudioPipeline(target_fps=60)
    pipeline60.process(temp_audio_file)
    
    files_initial = list(tmp_path.glob("*.json"))
    assert len(files_initial) == 1
    file_60 = files_initial[0]
    
    # Run with FPS 30
    pipeline30 = AudioPipeline(target_fps=30)
    pipeline30.process(temp_audio_file)
    
    files_secondary = list(tmp_path.glob("*.json"))
    # Should be 2 files total now (the old one + new one)
    assert len(files_secondary) == 2
    
    # Verify the new file is different from the old one
    new_files = [f for f in files_secondary if f != file_60]
    assert len(new_files) == 1
    
    # Run with different impact envelope
    from chromascope.core.polisher import EnvelopeParams
    pipeline_env = AudioPipeline(
        target_fps=60, 
        impact_envelope=EnvelopeParams(attack_ms=100.0)
    )
    pipeline_env.process(temp_audio_file)
    
    # Should be 3 files total
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_clear_cache(tmp_path, monkeypatch):
    """Test that clear_cache removes the cache directory."""
    monkeypatch.setattr(AudioPipeline, "_get_cache_dir", lambda self: tmp_path)
    
    # Create a dummy cache file
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "dummy.json").touch()
    
    pipeline = AudioPipeline()
    pipeline.clear_cache()
    
    # Directory should be empty (recreated but empty)
    assert len(list(tmp_path.glob("*"))) == 0


def test_cache_invalidation_on_version_change(temp_audio_file, tmp_path, monkeypatch):
    """Test that changing ANALYSIS_VERSION invalidates the cache."""
    monkeypatch.setattr(AudioPipeline, "_get_cache_dir", lambda self: tmp_path)
    
    # 1. Run with current version
    pipeline1 = AudioPipeline(target_fps=60)
    pipeline1.process(temp_audio_file)
    
    files_initial = list(tmp_path.glob("*.json"))
    assert len(files_initial) == 1
    initial_cache_file = files_initial[0]
    
    # 2. Mock a version change in the class
    monkeypatch.setattr(AudioPipeline, "ANALYSIS_VERSION", "9.9-test")
    
    pipeline2 = AudioPipeline(target_fps=60)
    # This should result in a cache MISS and call decompose
    with patch.object(pipeline2, 'decompose', wraps=pipeline2.decompose) as mock_decompose:
        pipeline2.process(temp_audio_file)
        assert mock_decompose.called
        
    # Should be 2 files total now
    files_final = list(tmp_path.glob("*.json"))
    assert len(files_final) == 2
    assert initial_cache_file in files_final
