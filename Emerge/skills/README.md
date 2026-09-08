# Emerge Skills

This directory contains built-in skills that extend Emerge's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `memory` | Two-layer memory system (always loaded) |
| `image` | Capture images and save to workspace |
| `attach` | Rule-based robot control (geometry-driven motion primitives) |
| `vla` | VLA policy control (natural language instructions) |
