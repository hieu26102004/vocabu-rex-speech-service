"""
Test configuration and fixtures
"""

import pytest
import asyncio
from typing import Generator


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "Hello world, how are you today?"


@pytest.fixture
def sample_words():
    """Sample words for testing"""
    return ["hello", "world", "pronunciation", "assessment"]