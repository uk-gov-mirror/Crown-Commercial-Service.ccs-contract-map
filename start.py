#!/usr/bin/env python3
"""
Entry point for the CCS Contract Map API server.

Usage:
    python start.py                    # Normal startup (uses existing venv)
    python start.py --setup            # Create venv, install deps, then start
    python start.py --install-only     # Only create venv and install deps
    python start.py --check            # Check if venv and deps are ready

Environment Variables:
    SETUP_VENV=true          # Auto-create venv and install on startup
    VENV_PATH=.venv          # Virtual environment path
    REQUIREMENTS_FILE=requirements.txt  # Requirements file
    SKIP_DEP_CHECK=false     # Skip dependency verification
"""

import os
import sys
import subprocess
import venv
import shutil
from pathlib import Path
from typing import Optional

# Ensure src is on the path early
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Virtual Environment Management

class VenvManager:
    """Manages virtual environment creation and dependency installation."""

    def __init__(
        self,
        repo_root: Path,
        venv_path: Optional[Path] = None,
        requirements_file: Optional[Path] = None,
    ):
        self.repo_root = repo_root
        self.venv_path = venv_path or repo_root / ".venv"
        self.requirements_file = requirements_file or repo_root / "requirements.txt"
        self.python_exe = self._get_python_executable()

    def _get_python_executable(self) -> Path:
        """Get the python executable path for the venv."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    def _get_pip_executable(self) -> Path:
        """Get the pip executable path for the venv."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"

    def venv_exists(self) -> bool:
        """Check if virtual environment exists and is valid."""
        return self.python_exe.exists()

    def create_venv(self, force: bool = False) -> bool:
        """
        Create a new virtual environment.

        Args:
            force: If True, delete existing venv first

        Returns:
            True if created successfully
        """
        if self.venv_exists() and not force:
            print(f"Virtual environment already exists at {self.venv_path}")
            return True

        if self.venv_exists() and force:
            print(f"Removing existing virtual environment...")
            shutil.rmtree(self.venv_path)

        print(f"Creating virtual environment at {self.venv_path}...")
        try:
            venv.create(self.venv_path, with_pip=True, upgrade_deps=True)
            print(f"Virtual environment created successfully")
            return True
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            return False

    def install_requirements(self, upgrade: bool = False) -> bool:
        """
        Install dependencies from requirements.txt.

        Args:
            upgrade: If True, upgrade packages to latest versions

        Returns:
            True if installation successful
        """
        if not self.requirements_file.exists():
            print(f"Requirements file not found: {self.requirements_file}")
            return False

        if not self.venv_exists():
            print(f"Virtual environment not found. Run create_venv() first.")
            return False

        pip_cmd = [str(self._get_pip_executable()), "install"]
        if upgrade:
            pip_cmd.append("--upgrade")
        pip_cmd.extend(["-r", str(self.requirements_file)])

        print(f"Installing dependencies from {self.requirements_file}...")
        print(f"   Command: {' '.join(pip_cmd)}")

        try:
            result = subprocess.run(
                pip_cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode == 0:
                print(f"Dependencies installed successfully")
                return True
            else:
                print(f"Failed to install dependencies:")
                print(result.stdout)
                print(result.stderr)
                return False
        except subprocess.TimeoutExpired:
            print(f"Installation timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"Installation failed: {e}")
            return False

    def check_dependencies(self) -> tuple[bool, list[str]]:
        """
        Check if all required packages are installed.

        Returns:
            Tuple of (all_satisfied, missing_packages)
        """
        if not self.venv_exists():
            return False, ["Virtual environment not found"]

        pip_cmd = [str(self._get_pip_executable()), "check"]
        try:
            result = subprocess.run(
                pip_cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True, []
            else:
                # Parse missing packages from output
                missing = []
                for line in result.stdout.splitlines():
                    if "not installed" in line or "not satisfied" in line:
                        missing.append(line.strip())
                return False, missing
        except Exception as e:
            return False, [f"Check failed: {e}"]

    def get_activation_script(self) -> str:
        """Get the shell command to activate the virtual environment."""
        if sys.platform == "win32":
            return f"{self.venv_path}\\Scripts\\activate"
        return f"source {self.venv_path}/bin/activate"


# Environment Configuration (.env file management)

class EnvManager:
    """Manages .env file for credentials and configuration."""

    REQUIRED_VARS = {
        "AZURE_OPENAI_ENDPOINT": {
            "description": "Azure OpenAI endpoint URL",
            "example": "https://your-resource.openai.azure.com/",
            "required": True,
        },
        "AZURE_OPENAI_KEY": {
            "description": "Azure OpenAI API key",
            "example": "your-api-key-here",
            "required": True,
        },
        "DEPLOYMENT_NAME": {
            "description": "Azure OpenAI deployment name",
            "example": "gpt-4o",
            "required": True,
        },
        "AZURE_OPENAI_API_VERSION": {
            "description": "Azure OpenAI API version",
            "example": "2024-02-01",
            "required": False,
            "default": "2024-02-01",
        },
    }

    OPTIONAL_VARS = {
        "REDIS_HOST": {
            "description": "Redis host for caching (optional)",
            "example": "redis",
            "required": False,
        },
        "MLFLOW_TRACKING_URI": {
            "description": "MLflow tracking URI (optional)",
            "example": "azureml://...",
            "required": False,
        },
        "MLFLOW_EXPERIMENT_NAME": {
            "description": "MLflow experiment name",
            "example": "ContractMap-Evaluation",
            "required": False,
            "default": "ContractMap-Evaluation",
        },
    }

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.env_path = repo_root / ".env"
        self.env_example_path = repo_root / ".env.example"

    def is_docker(self) -> bool:
        """Detect if running inside a Docker container."""
        # Check for .dockerenv file (most reliable)
        if Path("/.dockerenv").exists():
            return True
        # Check cgroup for docker
        try:
            with open("/proc/self/cgroup", "r") as f:
                if "docker" in f.read():
                    return True
        except Exception:
            pass
        # Check environment variable
        if os.getenv("RUNNING_IN_DOCKER", "").lower() == "true":
            return True
        return False

    def load_env(self) -> dict:
        """Load existing .env file into dict."""
        env_vars = {}
        if self.env_path.exists():
            with open(self.env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars

    def save_env(self, env_vars: dict) -> bool:
        """Save environment variables to .env file."""
        try:
            with open(self.env_path, "w") as f:
                f.write("# CCS Contract Map - Environment Configuration\n")
                f.write("# Generated by start.py setup\n\n")
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            return True
        except Exception as e:
            print(f"Failed to save .env: {e}")
            return False

    def get_missing_required(self, current: dict) -> list[str]:
        """Get list of required variables that are missing or empty."""
        missing = []
        for var, config in self.REQUIRED_VARS.items():
            if config["required"] and (var not in current or not current[var]):
                missing.append(var)
        return missing

    def prompt_for_credentials(self, current: dict) -> dict:
        """Interactively prompt user for missing credentials."""
        print("\n" + "=" * 60)
        print("Azure OpenAI Configuration Required")
        print("=" * 60)
        print("The API requires Azure OpenAI credentials to function.")
        print("You can find these in your Azure Portal > Azure OpenAI resource.")
        print()

        updated = current.copy()

        # Required variables
        for var, config in self.REQUIRED_VARS.items():
            if var in updated and updated[var]:
                print(f"{var} already configured")
                continue

            default = config.get("default", "")
            prompt_text = f"{var}"
            if config.get("example"):
                prompt_text += f" (e.g., {config['example']})"
            if default:
                prompt_text += f" [{default}]"
            prompt_text += ": "

            while True:
                value = input(prompt_text).strip()
                if not value and default:
                    value = default
                if value:
                    updated[var] = value
                    break
                print(f"   {var} is required. Please enter a value.")

        # Optional variables
        print("\nOptional configuration (press Enter to skip):")
        for var, config in self.OPTIONAL_VARS.items():
            if var in updated and updated[var]:
                continue

            default = config.get("default", "")
            prompt_text = f"{var}"
            if config.get("example"):
                prompt_text += f" (e.g., {config['example']})"
            if default:
                prompt_text += f" [{default}]"
            prompt_text += ": "

            value = input(prompt_text).strip()
            if not value and default:
                value = default
            if value:
                updated[var] = value

        print("=" * 60)
        return updated

    def ensure_configured(self, interactive: bool = True) -> bool:
        """
        Ensure .env file has all required credentials.

        Args:
            interactive: If True, prompt user for missing values.
                        If False (e.g., in Docker), skip prompting.

        Returns:
            True if all required credentials are present
        """
        # Skip interactive prompting in Docker
        if self.is_docker():
            print("🐳 Running in Docker - skipping interactive credential setup")
            print("   Ensure credentials are provided via environment variables or mounted .env")
            return True

        current = self.load_env()
        missing = self.get_missing_required(current)

        if not missing:
            print("All required credentials found in .env")
            return True

        if not interactive:
            print("Missing required credentials (non-interactive mode):")
            for var in missing:
                print(f"   - {var}")
            print("   Run with --setup to configure interactively, or set env vars manually.")
            return False

        print(f" Missing {len(missing)} required credential(s): {', '.join(missing)}")

        # Check if .env.example exists and offer to copy
        if self.env_example_path.exists() and not self.env_path.exists():
            print(f"\n📋 Found .env.example - copying as starting point...")
            import shutil
            shutil.copy(self.env_example_path, self.env_path)
            current = self.load_env()
            missing = self.get_missing_required(current)

        # Prompt for missing credentials
        updated = self.prompt_for_credentials(current)

        # Save
        if self.save_env(updated):
            print(f"\nCredentials saved to {self.env_path}")
            return True
        else:
            print(f"\nFailed to save credentials")
            return False



# Setup Functions


def setup_environment(
    force_venv: bool = False,
    upgrade_deps: bool = False,
    skip_check: bool = False,
    skip_env_prompt: bool = False,
) -> bool:
    """
    Complete environment setup: create venv, install deps, configure .env, verify.

    Args:
        force_venv: Recreate venv even if exists
        upgrade_deps: Upgrade packages to latest
        skip_check: Skip dependency verification
        skip_env_prompt: Skip interactive .env configuration

    Returns:
        True if setup successful
    """
    print("=" * 60)
    print("CCS Contract Map - Environment Setup")
    print("=" * 60)

    # Step 0: Configure .env credentials (unless in Docker or skipped)
    env_manager = EnvManager(REPO_ROOT)
    if not skip_env_prompt:
        if not env_manager.ensure_configured(interactive=True):
            print("Credential configuration incomplete")
            return False
    else:
        # In non-interactive mode, just verify required vars exist in environment
        if not env_manager.ensure_configured(interactive=False):
            print(" Some credentials missing - ensure they are set via environment variables")

    manager = VenvManager(REPO_ROOT)

    # Step 1: Create virtual environment
    if not manager.create_venv(force=force_venv):
        return False

    # Step 2: Install dependencies
    if not manager.install_requirements(upgrade=upgrade_deps):
        return False

    # Step 3: Verify installation
    if not skip_check:
        print("🔍 Verifying installation...")
        ok, missing = manager.check_dependencies()
        if ok:
            print("All dependencies satisfied")
        else:
            print(" Some dependencies may be missing:")
            for m in missing:
                print(f"   - {m}")
            # Don't fail on check - just warn

    print("=" * 60)
    print("Environment setup complete!")
    print(f"   To activate manually: {manager.get_activation_script()}")
    print("=" * 60)
    return True


def ensure_venv_activated() -> bool:
    """
    Ensure we're running inside the virtual environment.
    If not, re-execute with the venv python.

    Returns:
        True if we're in the correct venv (or setup succeeded)
    """
    manager = VenvManager(REPO_ROOT)

    # Already in the correct venv?
    if Path(sys.executable).resolve().is_relative_to(manager.venv_path.resolve()):
        return True

    # Venv exists but we're not using it - re-exec
    if manager.venv_exists():
        print(f"Re-executing with virtual environment python...")
        os.execv(str(manager.python_exe), [str(manager.python_exe)] + sys.argv)

    # No venv - check if auto-setup is enabled
    if os.getenv("SETUP_VENV", "false").lower() == "true":
        print("No virtual environment found. Auto-creating...")
        if setup_environment(skip_env_prompt=True):  # Non-interactive in auto mode
            # Re-exec with new venv
            os.execv(str(manager.python_exe), [str(manager.python_exe)] + sys.argv)
        else:
            print("Auto-setup failed")
            return False

    # No venv and no auto-setup - warn but continue (maybe global install)
    print(f" Not running in virtual environment ({manager.venv_path})")
    print(f"   Run 'python start.py --setup' to create one, or set SETUP_VENV=true")
    return True



# Main Entry Points


def get_app():
    """Import and return the FastAPI app instance."""
    from src.api.v3_endpoint import app
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info",
    is_production: bool = False,
):
    """Run the API server with Uvicorn."""
    import uvicorn

    if is_production:
        uvicorn.run(
            "start:get_app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True,
            factory=True,
        )
    else:
        uvicorn.run(
            "start:get_app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            factory=True,
        )


def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CCS Contract Map API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                    # Start server (uses existing venv)
  python start.py --setup            # Create venv, install deps, configure .env, then start
  python start.py --install-only     # Only setup environment, don't start
  python start.py --check            # Verify environment is ready
  SETUP_VENV=true python start.py    # Auto-setup on startup (non-interactive)
  RUNNING_IN_DOCKER=true python start.py  # Skip .env prompts in Docker
        """
    )

    # Setup commands
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create venv, install dependencies, configure .env, then start server",
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Only create venv, install dependencies, and configure .env, don't start server",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if environment is ready, exit with status code",
    )
    parser.add_argument(
        "--force-venv",
        action="store_true",
        help="Force recreate virtual environment",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade dependencies to latest versions",
    )
    parser.add_argument(
        "--skip-env-prompt",
        action="store_true",
        help="Skip interactive .env credential prompting (use existing env vars)",
    )

    # Server options
    parser.add_argument("--host", default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "5000")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("API_WORKERS", "1")))
    parser.add_argument("--reload", action="store_true", default=os.getenv("API_RELOAD", "false").lower() == "true")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    parser.add_argument("--production", action="store_true", default=os.getenv("ENVIRONMENT", "development").lower() == "production")

    args = parser.parse_args()

    # Handle setup commands
    if args.check:
        manager = VenvManager(REPO_ROOT)
        env_manager = EnvManager(REPO_ROOT)

        venv_ok = manager.venv_exists()
        deps_ok, missing = manager.check_dependencies() if venv_ok else (False, [])
        env_ok = len(env_manager.get_missing_required(env_manager.load_env())) == 0

        if venv_ok and deps_ok and env_ok:
            print("Environment ready")
            sys.exit(0)
        else:
            print("Environment incomplete:")
            if not venv_ok:
                print("   - Virtual environment not found")
            if not deps_ok:
                for m in missing:
                    print(f"   - {m}")
            if not env_ok:
                missing_env = env_manager.get_missing_required(env_manager.load_env())
                for m in missing_env:
                    print(f"   - Missing credential: {m}")
            sys.exit(1)

    if args.setup or args.install_only:
        success = setup_environment(
            force_venv=args.force_venv,
            upgrade_deps=args.upgrade,
            skip_env_prompt=args.skip_env_prompt,
        )
        if args.install_only:
            sys.exit(0 if success else 1)
        if not success:
            sys.exit(1)

    # Ensure we're in the right environment for server start
    if not ensure_venv_activated():
        if os.getenv("SETUP_VENV", "false").lower() != "true":
            print("Cannot start server without proper environment")
            sys.exit(1)

    # Start server
    print("=" * 60)
    print(f"Starting CCS Contract Map API")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Reload: {args.reload}")
    print(f"   Production: {args.production}")
    print("=" * 60)

    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        is_production=args.production,
    )


if __name__ == "__main__":
    main()