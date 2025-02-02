import inspect
import logging
from typing import List, Optional, Type

from pydantic import BaseModel
from rich import print as rprint
from rich.console import Console

from .base import BlockMetadata, FunctionalBlock

# Set up rich console and logging
console = Console()
logger = logging.getLogger(__name__)

def block(
    name: Optional[str] = None,
    description: Optional[str] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    version: str = "1.0",
    tags: Optional[List[str]] = None,
    validate: bool = True,
    debug: Optional[bool] = False
):
    """Decorator to create a functional block

    Example:
    -------
    @block(
        input_schema=InputModel,
        output_schema=OutputModel,
        tags=['preprocessing']
    )
    async def process_data(data: InputModel) -> OutputModel:
        ...

    """

    def _log_message(message: str, color: str = None) -> None:
        """Internal method for consistent logging"""
        if debug:
            if color:
                rprint(f"\n[{color}]{message}[/{color}]")
            else:
                rprint(f"\n{message}")

    def decorator(func):

        _log_message(f"Decorating function {func.__name__}", color="bold blue")

        # Extract function signature
        inspect.signature(func)

        # Get description from docstring if not provided
        block_description = description
        if not block_description and func.__doc__:
            block_description = func.__doc__.split("\n")[0].strip()
            _log_message("Block description not provided, using function docstring instead")
        block_description = block_description or f"Block: {func.__name__}"

        # Only create schemas if explicitly requested or if schemas are provided
        block_input_schema = input_schema
        block_output_schema = output_schema

        # Create metadata
        metadata = BlockMetadata(
            name=name or func.__name__,
            description=block_description,
            input_schema=block_input_schema,
            output_schema=block_output_schema,
            version=version,
            tags=tags or []
        )

        # Create and return block
        return FunctionalBlock(
            func=func,
            metadata=metadata,
            validate=validate
        )

    return decorator
