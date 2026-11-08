"""Shared code for the sarcasm-circuit project.

Modules
-------
model     load a Qwen model (3B for development, 7B for final results)
prompts   dataset -> token ids + regions; clean/corrupted pairs
metric    the score d = logit(Yes) - logit(No)
hooks     read and replace internal activations (residual, attention, MLP, per head)
patching  activation patching sweeps and the recovery metric R
utils     results folder on Google Drive and cached computation
"""
