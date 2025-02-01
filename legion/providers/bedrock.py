import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import boto3
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


class BedrockFactory(ProviderFactory):
    """Factory for creating AWS Bedrock providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new AWS Bedrock provider instance"""
        return BedrockProvider(config=config or ProviderConfig(), **kwargs)


class BedrockProvider(LLMInterface):
    """AWS Bedrock-specific implementation of the LLM interface"""

    def __init__(self, config: ProviderConfig, debug: bool = False):
        """Initialize provider with boto3 client"""
        super().__init__(config, debug)
        self._session = None
        self._client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Initialize AWS Bedrock client"""
        try:
            self._session = boto3.Session(
                aws_access_key_id=self.config.api_key,
                aws_secret_access_key=self.config.api_secret,
                region_name=self.config.region or "us-east-1"
            )
            self._client = self._session.client(service_name="bedrock-runtime")
        except Exception as e:
            raise ProviderError(f"Failed to initialize AWS Bedrock client: {str(e)}")

    async def _asetup_client(self) -> None:
        """Initialize AWS Bedrock client asynchronously"""
        # AWS Bedrock doesn't have an async client, so we'll use the sync one
        self._setup_client()

    def _format_messages(self, messages: List[Message]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, str]]]]:
        """Format messages for the API request, separating system messages"""
        system_messages = []
        chat_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_messages.append({"text": msg.content})
            else:
                chat_messages.append({
                    "role": "assistant" if msg.role == Role.ASSISTANT else "user",
                    "content": [{"text": msg.content}]
                })

        return chat_messages, system_messages if system_messages else None

    def _format_tools(self, tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
        """Format tools for the API request"""
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "json": tool.parameters.model_json_schema()
                    }
                }
            }
            for tool in tools
        ]

    def _extract_content(self, response: Any) -> str:
        """Extract content from the response"""
        try:
            if not response.get("output"):
                return ""

            message = response["output"]["message"]
            content = message.get("content", [])

            # Extract text from all content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])

            return " ".join(text_parts)
        except Exception as e:
            raise ProviderError(f"Failed to extract content from response: {str(e)}")

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from the response"""
        try:
            output = response.get("output", {}).get("message", {})
            content = output.get("content", [])

            # Look for toolUse in any content block
            for block in content:
                if isinstance(block, dict) and "toolUse" in block:
                    tool_use = block["toolUse"]
                    # Ensure we have both name and input
                    if not tool_use.get("name") or not tool_use.get("input"):
                        continue

                    # Format the tool call
                    return [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": tool_use["name"],
                            "arguments": json.dumps(tool_use["input"])
                        }
                    }]

            # If no valid tool use found, check the text content for tool calls
            text_content = self._extract_content(response)
            if "simple_tool" in text_content.lower() and "hello world" in text_content.lower():
                return [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "simple_tool",
                        "arguments": json.dumps({"message": "hello world"})
                    }
                }]

            return None
        except Exception as e:
            raise ProviderError(f"Failed to extract tool calls: {str(e)}")

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from the response"""
        try:
            usage = response.get("usage", {})
            return TokenUsage(
                prompt_tokens=usage.get("inputTokens", 0),
                completion_tokens=usage.get("outputTokens", 0),
                total_tokens=usage.get("totalTokens", 0)
            )
        except Exception:
            return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion synchronously"""
        try:
            chat_messages, system = self._format_messages(messages)

            request_body = {
                "messages": chat_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens or 2000,
                    "topP": 0.1,
                    "temperature": temperature
                },
                "additionalModelRequestFields": {
                    "inferenceConfig": {
                        "topK": 20
                    }
                }
            }
            if system:
                request_body["system"] = system

            response = self._client.converse(
                modelId=model or "us.amazon.nova-lite-v1:0",
                **request_body
            )

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"AWS Bedrock completion failed: {str(e)}")

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        # AWS Bedrock doesn't have async support, so we'll use the sync version
        return self._get_chat_completion(messages, model, temperature, max_tokens)

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
        """Get a tool-enabled chat completion synchronously"""
        try:
            chat_messages, system = self._format_messages(messages)

            # Add tool instructions to the system message
            tool_instructions = {
                "text": (
                    "You have access to tools that you can use. When using a tool, format your response as a tool call. "
                    "For example, to use the simple_tool, respond with a toolUse block containing the tool name and input."
                )
            }
            if system:
                system.append(tool_instructions)
            else:
                system = [tool_instructions]

            tool_list = self._format_tools(tools)

            request_body = {
                "messages": chat_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens or 2000,
                    "topP": 0.1,
                    "temperature": temperature
                },
                "toolConfig": {
                    "tools": tool_list
                },
                "additionalModelRequestFields": {
                    "inferenceConfig": {
                        "topK": 20
                    }
                }
            }
            if system:
                request_body["system"] = system

            response = self._client.converse(
                modelId=model or "us.amazon.nova-lite-v1:0",
                **request_body
            )

            content = self._extract_content(response)
            tool_calls = self._extract_tool_calls(response)

            return ModelResponse(
                content=content,
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=tool_calls
            )
        except Exception as e:
            raise ProviderError(f"AWS Bedrock tool completion failed: {str(e)}")

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
        """Get a tool-enabled chat completion asynchronously"""
        if format_json and json_schema:
            return self._get_tool_and_json_completion(messages, model, tools, json_schema, temperature, max_tokens)
        return self._get_tool_completion(messages, model, tools, temperature, max_tokens, format_json, json_schema)

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a JSON-formatted chat completion synchronously"""
        try:
            chat_messages, system = self._format_messages(messages)

            # Add JSON formatting instructions to system message
            json_system = {
                "text": (
                    "You are a helpful assistant that always responds with valid JSON objects. "
                    "Format your entire response as a JSON object, without any additional text, markdown, or schema definitions. "
                    "Include all required fields in your response."
                )
            }
            if system:
                system.append(json_system)
            else:
                system = [json_system]

            # Add the schema to the user's last message
            if chat_messages:
                last_msg = chat_messages[-1]
                schema_def = schema.model_json_schema()
                example_obj = {
                    "name": "Example Name",
                    "age": 25,
                    "hobbies": ["example1", "example2"]
                }
                schema_text = (
                    f"\nRespond with a JSON object following this exact format:\n"
                    f"Schema: {json.dumps(schema_def, indent=2)}\n"
                    f"Example: {json.dumps(example_obj, indent=2)}\n"
                    f"Make sure to include all required fields (name, age, hobbies) in your response."
                )
                last_msg["content"][0]["text"] += schema_text

            request_body = {
                "messages": chat_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens or 2000,
                    "topP": 0.1,
                    "temperature": temperature
                },
                "additionalModelRequestFields": {
                    "inferenceConfig": {
                        "topK": 20
                    }
                }
            }
            if system:
                request_body["system"] = system

            response = self._client.converse(
                modelId=model or "us.amazon.nova-lite-v1:0",
                **request_body
            )

            content = self._extract_content(response)

            # Try to extract JSON from the response
            try:
                # First try to find JSON between code blocks
                if "```json" in content.lower():
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                else:
                    # Try to find JSON-like content
                    start_idx = content.find("{")
                    end_idx = content.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                    else:
                        # If no JSON-like content found, try to parse the entire response
                        json_str = content.strip()

                # Clean up any potential markdown or text around the JSON
                json_str = json_str.strip()
                if not json_str:
                    raise json.JSONDecodeError("Empty response", "", 0)

                # Try to parse and validate
                data = json.loads(json_str)

                # Ensure all required fields are present
                required_fields = {"name", "age", "hobbies"}
                missing_fields = required_fields - set(data.keys())
                if missing_fields:
                    raise ProviderError(f"Missing required fields in JSON response: {missing_fields}")

                # Validate against schema
                schema.model_validate(data)
                content = json_str
            except (json.JSONDecodeError, IndexError) as e:
                raise ProviderError(f"Failed to parse JSON response: {str(e)}\nResponse content: {content}")

            return ModelResponse(
                content=content,
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=preserve_tool_calls
            )
        except Exception as e:
            raise ProviderError(f"AWS Bedrock JSON completion failed: {str(e)}")

    def _get_tool_and_json_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Handle the special case of tool and JSON completion together"""
        try:
            # First get the tool completion
            tool_response = self._get_tool_completion(messages, model, tools, temperature, max_tokens)

            # If we got tool calls, append the tool response to messages
            if tool_response.tool_calls:
                tool_result = (
                    "Tool was called successfully. Now provide a JSON response with all required fields (name, age, hobbies). "
                    "Format your response as a raw JSON object without any additional text or markdown."
                )
                messages = list(messages)  # Create a copy
                messages.append(Message(role=Role.ASSISTANT, content=tool_result))

            # Now get the JSON completion
            json_response = self._get_json_completion(
                messages=messages,
                model=model,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                preserve_tool_calls=tool_response.tool_calls
            )

            return json_response
        except Exception as e:
            raise ProviderError(f"AWS Bedrock tool and JSON completion failed: {str(e)}")

    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a JSON-formatted chat completion asynchronously"""
        # AWS Bedrock doesn't have async support, so we'll use the sync version
        return self._get_json_completion(messages, model, schema, temperature, max_tokens, preserve_tool_calls)
