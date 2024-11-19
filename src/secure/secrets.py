"""Secure secrets management using system keyring."""

from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import keyring
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manages secure access to API keys and other sensitive credentials."""

    def __init__(self, service_name: str = None):
        """Initialize SecretsManager.

        Args:
            service_name: Name of the service in keyring. Defaults to directory name.
        """
        self._load_environment()
        if service_name is None:
            # Use the project directory name as default service name
            service_name = Path(__file__).parent.parent.parent.name
        self.service_name = service_name

    def _load_environment(self):
        """Load environment variables from .env file if it exists."""
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    @lru_cache()
    def get_secret(self, key_name: str) -> Optional[str]:
        """Get a secret from the system keyring or environment variables.

        Args:
            key_name: Name of the secret to retrieve

        Returns:
            The secret value or None if not found
        """
        # First try to get from keyring
        secret = keyring.get_password(self.service_name, key_name)

        # If not in keyring, try environment variable
        if not secret:
            secret = os.getenv(key_name)

            # If found in env, store it in keyring for future use
            if secret:
                self.set_secret(key_name, secret)
                # Remove from environment after storing in keyring
                del os.environ[key_name]

        return secret

    def set_secret(self, key_name: str, value: str) -> None:
        """Store a secret in the system keyring.

        Args:
            key_name: Name of the secret
            value: Value to store
        """
        keyring.set_password(self.service_name, key_name, value)
        # Clear the cache so next get_secret call fetches fresh value
        self.get_secret.cache_clear()

    def delete_secret(self, key_name: str) -> None:
        """Delete a secret from the system keyring.

        Args:
            key_name: Name of the secret to delete
        """
        try:
            keyring.delete_password(self.service_name, key_name)
            # Clear the cache after deletion
            self.get_secret.cache_clear()
        except keyring.errors.PasswordDeleteError:
            logger.warning(f"Secret {key_name} not found in keyring")


# Create a singleton instance
secrets = SecretsManager()
