import asyncio
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from legion import block
from legion.blocks.base import BlockError, BlockMetadata, FunctionalBlock, ValidationError


# Test Models
class SimpleInput(BaseModel):
    value: str

class SimpleOutput(BaseModel):
    result: str

class ComplexInput(BaseModel):
    numbers: List[int]
    text: str
    optional_field: Optional[str] = None

class ComplexOutput(BaseModel):
    sum: int
    processed_text: str
    items_count: int

# Test Functions
def sync_processor(data: SimpleInput) -> Dict[str, str]:
    """Simple sync processor"""
    return {"result": f"Processed: {data.value}"}

async def async_processor(data: SimpleInput) -> Dict[str, str]:
    """Simple async processor"""
    await asyncio.sleep(0.1)
    return {"result": f"Async Processed: {data.value}"}

def complex_processor(data: ComplexInput) -> Dict[str, Any]:
    """Complex data processor"""
    return {
        "sum": sum(data.numbers),
        "processed_text": data.text.upper(),
        "items_count": len(data.numbers)
    }

# Test Fixtures
@pytest.fixture
def simple_metadata():
    return BlockMetadata(
        name="test_block",
        description="A test block",
        input_schema=SimpleInput,
        output_schema=SimpleOutput,
        version="1.0",
        tags=["test"]
    )

@pytest.fixture
def sync_block(simple_metadata):
    return FunctionalBlock(
        func=sync_processor,
        metadata=simple_metadata
    )

@pytest.fixture
def async_block(simple_metadata):
    return FunctionalBlock(
        func=async_processor,
        metadata=simple_metadata
    )

# Metadata Tests
def test_block_metadata():
    """Test block metadata creation and properties"""
    metadata = BlockMetadata(
        name="test",
        description="test description"
    )

    assert metadata.name == "test"
    assert metadata.description == "test description"
    assert metadata.version == "1.0"  # default value
    assert metadata.tags == []  # default value
    assert metadata.input_schema is None
    assert metadata.output_schema is None

def test_block_metadata_with_schemas():
    """Test block metadata with schemas"""
    metadata = BlockMetadata(
        name="test",
        description="test description",
        input_schema=SimpleInput,
        output_schema=SimpleOutput,
        version="2.0",
        tags=["test", "example"]
    )

    assert metadata.input_schema == SimpleInput
    assert metadata.output_schema == SimpleOutput
    assert metadata.version == "2.0"
    assert metadata.tags == ["test", "example"]

# Block Tests
def test_block_initialization(sync_block):
    """Test block initialization"""
    assert sync_block.func == sync_processor
    assert sync_block.validate is True
    assert sync_block.is_async is False
    assert isinstance(sync_block.metadata, BlockMetadata)

def test_async_block_initialization(async_block):
    """Test async block initialization"""
    assert async_block.func == async_processor
    assert async_block.validate is True
    assert async_block.is_async is True
    assert isinstance(async_block.metadata, BlockMetadata)

@pytest.mark.asyncio
async def test_sync_block_execution(sync_block):
    """Test synchronous block execution"""
    result = await sync_block(SimpleInput(value="test"))
    assert isinstance(result, SimpleOutput)
    assert result.result == "Processed: test"

@pytest.mark.asyncio
async def test_async_block_execution(async_block):
    """Test asynchronous block execution"""
    result = await async_block(SimpleInput(value="test"))
    assert isinstance(result, SimpleOutput)
    assert result.result == "Async Processed: test"

def test_block_validation():
    """Test input/output validation"""
    @block(
        input_schema=SimpleInput,
        output_schema=SimpleOutput
    )
    def validator(data: SimpleInput) -> Dict[str, str]:
        return {"result": f"Validated: {data.value}"}

    # Test valid input
    result = asyncio.run(validator(SimpleInput(value="test")))
    assert isinstance(result, SimpleOutput)
    assert result.result == "Validated: test"

    # Test invalid input
    with pytest.raises(ValidationError):
        asyncio.run(validator({"wrong_field": "test"}))

    # Test invalid output schema
    @block(
        input_schema=SimpleInput,
        output_schema=ComplexOutput
    )
    def invalid_output(data: SimpleInput) -> str:
        return "Invalid output type"

    with pytest.raises(ValidationError):
        asyncio.run(invalid_output(SimpleInput(value="test")))

def test_block_without_validation():
    """Test block execution without validation"""
    @block(validate=False)
    def no_validation(data: Any) -> Any:
        return data

    # Should accept any input
    result = asyncio.run(no_validation({"any": "data"}))
    assert result == {"any": "data"}

    result = asyncio.run(no_validation("string data"))
    assert result == "string data"

def test_complex_block():
    """Test block with complex input/output"""
    @block(
        input_schema=ComplexInput,
        output_schema=ComplexOutput
    )
    def process_complex(data: ComplexInput) -> Dict[str, Any]:
        return complex_processor(data)

    input_data = ComplexInput(
        numbers=[1, 2, 3, 4],
        text="test",
        optional_field="optional"
    )

    result = asyncio.run(process_complex(input_data))
    assert isinstance(result, ComplexOutput)
    assert result.sum == 10
    assert result.processed_text == "TEST"
    assert result.items_count == 4

def test_generic_type_validation():
    """Test validation with generic types"""
    @block(input_schema=List[int])
    def sum_numbers(numbers: List[int]) -> int:
        return sum(numbers)

    # Valid input
    result = asyncio.run(sum_numbers([1, 2, 3]))
    assert result == 6

    # Invalid input
    with pytest.raises(ValidationError):
        asyncio.run(sum_numbers(["1", "2", "3"]))  # strings instead of ints

    with pytest.raises(ValidationError):
        asyncio.run(sum_numbers("not a list"))  # not a list

def test_block_error_handling():
    """Test error handling in blocks"""
    @block(input_schema=SimpleInput)
    def error_block(data: SimpleInput) -> str:
        raise ValueError("Test error")

    with pytest.raises(BlockError) as exc_info:
        asyncio.run(error_block(SimpleInput(value="test")))
    assert "Test error" in str(exc_info.value)

def test_block_descriptor_protocol():
    """Test descriptor protocol for instance binding"""
    class BlockContainer:
        @block(input_schema=SimpleInput, output_schema=SimpleOutput)
        def process(self, data: SimpleInput) -> Dict[str, str]:
            return {"result": f"Instance processed: {data.value}"}

    container = BlockContainer()
    result = asyncio.run(container.process(SimpleInput(value="test")))
    assert isinstance(result, SimpleOutput)
    assert result.result == "Instance processed: test"

def test_block_decorator_customization():
    """Test block decorator customization"""
    @block(
        name="custom_block",
        description="Custom description",
        version="2.0",
        tags=["custom", "test"],
        validate=True
    )
    def custom_block(data: str) -> str:
        return f"Custom: {data}"

    assert custom_block.metadata.name == "custom_block"
    assert custom_block.metadata.description == "Custom description"
    assert custom_block.metadata.version == "2.0"
    assert custom_block.metadata.tags == ["custom", "test"]
    assert custom_block.validate is True

def test_block_docstring_description():
    """Test using docstring as block description"""
    @block()
    def docstring_block(data: str) -> str:
        """This is a test description"""
        return data

    assert docstring_block.metadata.description == "This is a test description"

@pytest.mark.asyncio
async def test_block_concurrent_execution():
    """Test concurrent execution of blocks"""
    @block()
    async def delayed_block(delay: float) -> float:
        await asyncio.sleep(delay)
        return delay

    # Execute blocks concurrently
    delays = [0.1, 0.2, 0.3]
    tasks = [delayed_block(d) for d in delays]
    results = await asyncio.gather(*tasks)

    assert results == delays

if __name__ == "__main__":
    pytest.main(["-v"])
