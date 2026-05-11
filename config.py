import os
from pathlib import Path

def load_env_file(path: str = ".env") -> None:
    # Use an absolute path if possible, or assume it's in the current working directory
    # For now, let's look in the current directory and the parent directory
    env_path = Path(path)
    if not env_path.exists():
        # Try looking one level up if not found (in case running from scripts/)
        env_path = Path(__file__).parent / path
        if not env_path.exists():
            return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value

def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        # Re-try loading just in case
        load_env_file()
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
    return value
