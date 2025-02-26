"""Legion: A provider-agnostic framework for building AI agent systems"""

__version__ = "0.1.5"

# Core interfaces
from legion.interface.decorators import tool, param, schema, output_schema, system_prompt

# Core agent system
from legion.agents.decorators import agent

# Groups
from legion.groups.decorators import chain, team, leader

# Blocks
from legion.blocks.decorators import block

# Provider management

# Error types
