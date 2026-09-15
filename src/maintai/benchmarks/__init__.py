"""Bounded public benchmark dataset adapters (read-only data slices).

Each benchmark is a self-contained vertical slice: metadata, a deterministic
parser, and (where relevant) a download/extract + prepare path. They never
train, evaluate, or call the API — those are later slices.
"""
