"""
Cascade module — the production inference pipeline.
Routes low-confidence predictions from the Base Student to the Corrector.
"""

from teascoFT.cascade.pipeline import CascadePipeline

__all__ = ["CascadePipeline"]
