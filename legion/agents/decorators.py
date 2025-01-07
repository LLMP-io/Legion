from typing import Optional, Type, Dict, Any, List, Union
from functools import wraps
import inspect
import asyncio
import logging

from legion.agents.base import Agent
from legion.interface.tools import BaseTool
from legion.interface.schemas import SystemPrompt, SystemPromptSection, Message, ModelResponse
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

def agent(
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[BaseTool]] = None,  # Allow tools to be passed directly
    **kwargs
):
    """Decorator for creating agent classes"""
    
    # Validate temperature
    if temperature < 0 or temperature > 1:
        raise ValueError("Temperature must be between 0 and 1")
    
    def decorator(cls: Type) -> Type:
        logger.debug(f"Decorating class {cls.__name__}")
        logger.debug(f"Original class bases: {cls.__bases__}")
        
        # Get system prompt from decorator or fallback to docstring
        if system_prompt is not None:  # Check decorator param first
            prompt_obj = SystemPrompt(static_prompt=system_prompt) if isinstance(system_prompt, str) else system_prompt
        elif hasattr(cls, '_system_prompt'):  # Then check for dynamic system prompt
            prompt_obj = cls._system_prompt
        elif cls.__doc__:  # Finally fallback to docstring
            prompt_obj = SystemPrompt(sections=[SystemPromptSection(
                content=cls.__doc__,
                is_dynamic=False
            )])
        else:
            prompt_obj = SystemPrompt(sections=[])  # Empty prompt if nothing provided
        
        # Create configuration
        config = {
            "model": model,
            "temperature": temperature,
            "system_prompt": prompt_obj,
            **kwargs
        }
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
            
        # Store original __init__ if it exists
        original_init = getattr(cls, '__init__', None)
        if original_init is object.__init__:
            original_init = None
        
        def __init__(self, *args, **kwargs):
            logger.debug(f"Initializing {cls.__name__} instance")
            logger.debug(f"Instance type: {type(self)}")
            logger.debug(f"Instance bases: {type(self).__bases__}")
            
            # Initialize Agent with config and proper name
            agent_config = {
                **config,
                "name": cls.__name__  # Always use the class name
            }
            
            logger.debug("Calling Agent.__init__")
            Agent.__init__(self, **agent_config)
            logger.debug("Agent.__init__ completed")
            
            # Initialize tools list
            self._tools = []
            
            # Get tools from class attributes with @tool decorator
            for attr_name, attr in inspect.getmembers(cls):
                if hasattr(attr, '__tool__'):
                    logger.debug(f"Found tool attribute: {attr_name}")
                    tool = attr.__tool_instance__
                    if tool:
                        logger.debug(f"Binding tool {tool.name} to instance")
                        self._tools.append(tool.bind_to(self))
                elif isinstance(attr, BaseTool):
                    logger.debug(f"Found BaseTool instance: {attr_name}")
                    self._tools.append(attr.bind_to(self))
            
            # Add tools passed to decorator
            if tools:
                logger.debug("Adding tools from decorator")
                for tool in tools:
                    logger.debug(f"Binding external tool {tool.name} to instance")
                    self._tools.append(tool.bind_to(self))
            
            # Get tools from constructor kwargs
            constructor_tools = kwargs.pop('tools', [])
            if constructor_tools:
                logger.debug("Adding tools from constructor kwargs")
                for tool in constructor_tools:
                    logger.debug(f"Binding constructor tool {tool.name} to instance")
                    self._tools.append(tool.bind_to(self))
            
            # Call the original class's __init__ if it exists
            if original_init:
                logger.debug("Calling original __init__")
                original_init(self, *args, **kwargs)
            
            if self.debug:
                logger.debug(f"Registered tools: {[t.name for t in self._tools]}")
        
        # Create new class attributes
        attrs = {
            '__init__': __init__,
            '__module__': cls.__module__,
            '__qualname__': cls.__qualname__,
            '__doc__': cls.__doc__,
            '_tools': [],  # Initialize class-level tools list
            '__agent_decorator__': True  # Mark as agent class
        }
        
        # Copy over class attributes and methods
        for attr_name, attr in cls.__dict__.items():
            if not attr_name.startswith('__'):
                attrs[attr_name] = attr
        
        # Create the new class with proper inheritance
        bases = (Agent,)
        if cls.__bases__ != (object,):
            bases = bases + tuple(b for b in cls.__bases__ if b != object)
        
        logger.debug(f"Creating new class with bases: {bases}")    
        AgentClass = type(cls.__name__, bases, attrs)
        logger.debug(f"Created class {AgentClass.__name__} with MRO: {AgentClass.__mro__}")
        
        # Copy over any class-level tools
        if hasattr(cls, '_tools'):
            logger.debug("Copying class-level tools")
            AgentClass._tools = cls._tools.copy()
        
        return AgentClass
    
    return decorator