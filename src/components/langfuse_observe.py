from functools import wraps
from typing import Callable, Any

from langfuse import Langfuse

from src import logger
from src.secure.secrets import secrets


def langfuse_observe(span_name: str = None, metadata: dict = None) -> Callable:
    """
    Decorator for observing function execution with Langfuse.
    
    Args:
        span_name: Name for the Langfuse span
        metadata: Additional metadata to log
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Initialize Langfuse client
                langfuse = Langfuse(
                    public_key=secrets.get_secret("LANGFUSE_PUBLIC_KEY"),
                    secret_key=secrets.get_secret("LANGFUSE_SECRET_KEY"),
                    host=secrets.get_secret("LANGFUSE_HOST")
                )
                
                # Create trace and span
                trace = langfuse.trace(name=span_name or func.__name__)
                span = trace.span(name=span_name or func.__name__)
                
                # Add metadata if provided
                if metadata:
                    span.metadata = metadata
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Mark span as successful
                span.end(status="success")
                trace.end(status="success")
                
                return result
                
            except Exception as e:
                logger.error("Error in function execution", exc_info=True)
                
                # If Langfuse is initialized, log the error
                if 'span' in locals():
                    span.end(status="error", statusMessage=str(e))
                if 'trace' in locals():
                    trace.end(status="error", statusMessage=str(e))
                
                # Re-raise the exception
                raise
                
            finally:
                # Ensure Langfuse client is closed
                if 'langfuse' in locals():
                    langfuse.flush()
                    
        return wrapper
    return decorator
