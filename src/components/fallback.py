"""
Simple fallback response when no information is available.
"""

from typing import Dict, Any


def generate_fallback(state: Dict[str, Any]) -> Dict[str, Any]:
    """Set a default fallback response when no information is available."""
    state.response = "I apologize, but I don't have enough information to answer your question. Please try rephrasing or asking something else."
    return state