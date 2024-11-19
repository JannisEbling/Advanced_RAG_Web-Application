import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.secure.secrets import secrets
import os
from dotenv import load_dotenv


def init_secrets():
    """Initialize secrets from .env file into system keyring"""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        print(f"No .env file found at {env_path}!")
        return

    # Load environment variables
    load_dotenv(env_path)

    # List of secret keys to transfer
    secret_keys = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
    ]

    for key in secret_keys:
        value = os.getenv(key)
        if value:
            secrets.set_secret(key, value)
            print(f"Stored {key} in system keyring")

    # Create a template .env file
    template_content = """# API Keys are stored securely in the system keyring
# Run scripts/init_secrets.py to initialize
# Required keys:
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
"""

    # Backup original .env
    if env_path.exists():
        backup_path = env_path.with_suffix(".env.backup")
        env_path.rename(backup_path)
        print(f"Original .env backed up to {backup_path}")

    # Write template .env
    with open(env_path, "w") as f:
        f.write(template_content)

    print("\nSecrets have been securely stored in system keyring.")
    print("Original .env has been backed up and replaced with a template.")
    print("You can now safely delete the .env.backup file.")


if __name__ == "__main__":
    init_secrets()
