"""Naju v1 — reference implementation of the Naju model.

Naju is a natively discrete gated state-space model: decoupled forget/input
sigmoid gates drive a diagonal recurrence (no ZOH discretization), read out
through a 1/sqrt(d_state)-normalized state readout plus a learnable
feedthrough, and gated by a GLU-style output gate. See README.md for the
equations, options, and scan backends.
"""
from naju.mixer import NajuMixer, NajuBlock, RMSNorm
from naju.lm import NajuLM
from naju.scan import naju_scan, naju_scan_reference, resolve_backend
from naju.optim import build_optimizer

__all__ = ["NajuMixer", "NajuBlock", "RMSNorm", "NajuLM",
           "naju_scan", "naju_scan_reference", "resolve_backend",
           "build_optimizer"]
