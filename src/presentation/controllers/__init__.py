"""
Controllers package initialization - exports all routers
"""

from .phonemization_controller import phonemization_router
from .health_controller import health_router

# Export routers for main.py
speech_router = phonemization_router

__all__ = [
    "phonemization_router",
    "health_router", 
    "speech_router"
]