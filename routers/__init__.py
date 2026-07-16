"""Lesson 7 — dry-run routers.

Each router exposes `select(question, category) -> RouterChoice`, returning WHICH
model it would pick (and, for routers that can, whether to ground) WITHOUT running
the target model. The comparison is descriptive: what does each router's selection
gravitate toward, on two axes — model strength and grounding.
"""
