# File: llm_kit/providers/deepseek.py

import json
from typing import Any, Dict, List, Optional, Sequence, Type

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
from ..interface.schemas import (
    Message,
    ModelResponse,
    ProviderConfig,
    Role,
    TokenUsage,
)
from ..interface.tools import BaseTool
from .factory import ProviderFactory


class DeepSeekFactory(ProviderFactory):
    """Factory for creating DeepSeek providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new DeepSeek provider instance"""
        return DeepSeekProvider(config=config or ProviderConfig(), **kwargs)

class DeepSeekProvider(LLMInterface):
    """DeepSeek-specific implementation of the LLM interface"""

    def __init__(self, config: ProviderConfig, debug: bool = False):
        """Initialize provider with both sync and async clients"""
        super().__init__(config, debug)
        self._async_client = None  # Initialize async client lazily

    def _setup_client(self) -> None:
        """Initialize DeepSeek client"""
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.deepseek.com",
                organization=self.config.organization_id,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize DeepSeek client: {str(e)}")

    async def _asetup_client(self) -> None:
        """Initialize async DeepSeek client"""
        try:
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.deepseek.com",
                organization=self.config.organization_id,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize async DeepSeek client: {str(e)}")

    async def _ensure_async_client(self) -> None:
        """Ensure async client is initialized"""
        if self._async_client is None:
            await self._asetup_client()

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        try:
            await self._ensure_async_client()
            response = await self._async_client.chat.completions.create(
                model=model,
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek async completion failed: {str(e)}")

    async def _aget_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get completion with tool usage asynchronously"""
        await self._ensure_async_client()
        current_messages = list(messages)
        all_tool_calls = []

        try:
            # First phase: Use tools
            while True:
                if self.debug:
                    print("\n🔄 Making async OpenAI API call:")
                    print(f"Messages count: {len(current_messages)}")
                    print("Tools:", [t.name for t in tools])

                # Convert messages and tools to dict format
                message_dicts = self._format_messages(current_messages)
                tool_dicts = [t.model_dump() for t in tools]

                try:
                    response = await self._async_client.chat.completions.create(
                        model=model,
                        messages=message_dicts,
                        tools=tool_dicts,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as api_error:
                    if self.debug:
                        print(f"\n❌ Async API call failed: {str(api_error)}")
                    raise

                choice = response.choices[0]
                content = choice.message.content or ""

                # Process tool calls if any
                if choice.message.tool_calls:
                    # First add the assistant's message with tool calls
                    tool_call_data = []
                    for tool_call in choice.message.tool_calls:
                        call_data = {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        tool_call_data.append(call_data)

                    # Add the assistant's message with tool calls
                    current_messages.append(Message(
                        role=Role.ASSISTANT,
                        content=content,
                        tool_calls=tool_call_data
                    ))

                    # Process each tool call
                    for tool_call in choice.message.tool_calls:
                        tool = next(
                            (t for t in tools if t.name == tool_call.function.name),
                            None
                        )

                        if tool:
                            args = json.loads(tool_call.function.arguments)
                            result = await tool.arun(**args)  # Use async tool call

                            if self.debug:
                                print(f"Tool {tool.name} returned: {result}")

                            # Add the tool's response
                            current_messages.append(Message(
                                role=Role.TOOL,
                                content=json.dumps(result) if isinstance(result, dict) else str(result),
                                tool_call_id=tool_call.id,
                                name=tool_call.function.name
                            ))

                            # Store tool call for final response
                            call_data = next(
                                c for c in tool_call_data
                                if c["id"] == tool_call.id
                            )
                            call_data["result"] = json.dumps(result) if isinstance(result, dict) else str(result)
                            all_tool_calls.append(call_data)
                    continue

                # No more tool calls - get final response
                if format_json and json_schema:
                    json_response = await self._aget_json_completion(
                        messages=current_messages,
                        model=model,
                        schema=json_schema,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        preserve_tool_calls=all_tool_calls if all_tool_calls else None
                    )
                    return json_response

                return ModelResponse(
                    content=content,
                    raw_response=self._response_to_dict(response),
                    tool_calls=all_tool_calls if all_tool_calls else None,
                    usage=self._extract_usage(response)
                )

        except Exception as e:
            raise ProviderError(f"OpenAI async tool completion failed: {str(e)}")

    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON asynchronously"""
        try:
            await self._ensure_async_client()

            # Get generic JSON formatting prompt
            formatting_prompt = self._get_json_formatting_prompt(schema, messages[-1].content)

            # Create messages for OpenAI
            openai_messages = [
                {"role": "system", "content": formatting_prompt}
            ]

            # Add remaining messages, skipping system
            openai_messages.extend([
                msg.model_dump() for msg in messages
                if msg.role != Role.SYSTEM
            ])

            response = await self._async_client.chat.completions.create(
                model=model,
                messages=openai_messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Validate against schema
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                schema.model_validate(data)
            except Exception as e:
                raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=preserve_tool_calls
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek JSON completion failed: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format"""
        openai_messages = []

        # Add system messages first (deduplicated)
        system_messages = [msg for msg in messages if msg.role == Role.SYSTEM]
        if system_messages:
            # Use the last system message if multiple exist
            openai_messages.append({
                "role": "system",
                "content": system_messages[-1].content
            })

        # Add remaining messages in order
        for msg in messages:
            if msg.role != Role.SYSTEM:
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content
                }

                # Add tool calls if present
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls

                # Add tool call id and name if present
                if msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    msg_dict["name"] = msg.name

                openai_messages.append(msg_dict)

        return openai_messages

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek completion failed: {str(e)}")

    def _get_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get completion with tool usage"""
        current_messages = list(messages)
        all_tool_calls = []

        try:
            # First phase: Use tools
            while True:
                if self.debug:
                    print("\n🔄 Making DeepSeek API call:")
                    print(f"Messages count: {len(current_messages)}")
                    print("Tools:", [t.name for t in tools])

                # Convert messages and tools to dict format
                message_dicts = self._format_messages(current_messages)
                tool_dicts = [t.model_dump() for t in tools]

                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=message_dicts,
                        tools=tool_dicts,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as api_error:
                    if self.debug:
                        print(f"\n❌ API call failed: {str(api_error)}")
                    raise

                choice = response.choices[0]
                content = choice.message.content or ""

                # Process tool calls if any
                if choice.message.tool_calls:
                    # First add the assistant's message with tool calls
                    tool_call_data = []
                    for tool_call in choice.message.tool_calls:
                        call_data = {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        tool_call_data.append(call_data)

                    # Add the assistant's message with tool calls
                    current_messages.append(Message(
                        role=Role.ASSISTANT,
                        content=content,
                        tool_calls=tool_call_data
                    ))

                    # Process each tool call
                    for tool_call in choice.message.tool_calls:
                        tool = next(
                            (t for t in tools if t.name == tool_call.function.name),
                            None
                        )

                        if tool:
                            args = json.loads(tool_call.function.arguments)
                            # Use sync run method instead of async call
                            result = tool.run(**args)

                            if self.debug:
                                print(f"Tool {tool.name} returned: {result}")

                            # Add the tool's response
                            current_messages.append(Message(
                                role=Role.TOOL,
                                content=json.dumps(result) if isinstance(result, dict) else str(result),
                                tool_call_id=tool_call.id,
                                name=tool_call.function.name
                            ))

                            # Store tool call for final response
                            call_data = next(
                                c for c in tool_call_data
                                if c["id"] == tool_call.id
                            )
                            call_data["result"] = json.dumps(result) if isinstance(result, dict) else str(result)
                            all_tool_calls.append(call_data)
                    continue

                # No more tool calls - get final response
                if format_json and json_schema:
                    json_response = self._get_json_completion(
                        messages=current_messages,
                        model=model,
                        schema=json_schema,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        preserve_tool_calls=all_tool_calls if all_tool_calls else None
                    )
                    return json_response

                return ModelResponse(
                    content=content,
                    raw_response=self._response_to_dict(response),
                    tool_calls=all_tool_calls if all_tool_calls else None,
                    usage=self._extract_usage(response)
                )

        except Exception as e:
            raise ProviderError(f"DeepSeek tool completion failed: {str(e)}")

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON"""
        try:
            # Get generic JSON formatting prompt
            formatting_prompt = self._get_json_formatting_prompt(schema, messages[-1].content)

            # Create messages for OpenAI
            openai_messages = [
                {"role": "system", "content": formatting_prompt}
            ]

            # Add remaining messages, skipping system
            openai_messages.extend([
                msg.model_dump() for msg in messages
                if msg.role != Role.SYSTEM
            ])

            response = self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Validate against schema
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                schema.model_validate(data)
            except Exception as e:
                raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=preserve_tool_calls
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek JSON completion failed: {str(e)}")

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from OpenAI response"""
        if not hasattr(response.choices[0].message, "tool_calls"):
            return None
        if not response.choices[0].message.tool_calls:
            return None

        tool_calls = []
        for tool_call in response.choices[0].message.tool_calls:
            call_data = {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }

            # Add tool call result if available
            if hasattr(tool_call, "result"):
                call_data["result"] = tool_call.result
            elif hasattr(tool_call.function, "result"):
                call_data["result"] = tool_call.function.result

            tool_calls.append(call_data)

        return tool_calls

    def _extract_content(self, response: Any) -> str:
        """Extract content from OpenAI response"""
        return response.choices[0].message.content or ""

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from response"""
        usage = response.usage
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert OpenAI response to dictionary"""
        return {
            "id": response.id,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            for tool_call in (choice.message.tool_calls or [])
                        ] if choice.message.tool_calls else None
                    }
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "created": response.created
        }

    def complete(
        self,
        messages: List[Message],
        model: str = "deepseek-chat",
        tools: Optional[Sequence[BaseTool]] = None,
        temperature: float = 1.0,
        response_schema: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a completion with optional tools and JSON formatting"""
        if self.debug:
            print("\n🔌 DeepSeek Provider:")
            print(f"Model: {model}")
            print(f"Temperature: {temperature}")
            print(f"Tools: {[t.name for t in (tools or [])]}")
            print(f"Response Schema: {response_schema.__name__ if response_schema else 'None'}")
            print("\nMessages:")
            for msg in messages:
                print(f"{msg.role}: {msg.content[:100]}...")

        try:
            if tools:
                if self.debug:
                    print("\n🛠️ Making tool completion request...")
                    print("Tool schemas:")
                    for tool in tools:
                        print(f"\n{tool.name}:")
                        print(json.dumps(tool.get_schema(), indent=2))

                return self._get_tool_completion(
                    messages=messages,
                    model=model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    format_json=bool(response_schema),
                    json_schema=response_schema
                )
            elif response_schema:
                if self.debug:
                    print("\n📋 Making JSON completion request...")
                    print("Schema:")
                    print(json.dumps(response_schema.model_json_schema(), indent=2))

                return self._get_json_completion(
                    messages=messages,
                    model=model,
                    schema=response_schema,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                if self.debug:
                    print("\n💬 Making basic completion request...")

                return self._get_chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        except Exception as e:
            if self.debug:
                print(f"\n❌ Provider error: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
            raise
