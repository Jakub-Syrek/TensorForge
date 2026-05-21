"""Vendored InstantX FLUX IP-Adapter custom code.

The three modules in this package (``attention_processor.py``,
``pipeline_flux_ipa.py``, ``transformer_flux.py``) are taken verbatim
from ``InstantX/FLUX.1-dev-IP-Adapter`` on Hugging Face (snapshot
``e44c6d889c951cac03ac806991e8d46c9ce1ddba``). They implement
InstantX's custom FLUX pipeline that integrates the IP-Adapter weights
via specialised attention processors — diffusers' own
``pipe.load_ip_adapter()`` doesn't speak this layout, so embedding the
upstream code is the only way to use the weights without rewriting
the inference loop from scratch.

Upstream license: see InstantX's repo (BSD-style at time of vendoring).
We add no modifications to the .py files themselves; the wrapper that
turns them into a TensorForge-friendly generator lives in
``backend/ipa.py``.
"""
