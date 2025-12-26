"""
Main entry point for running wikipedia_agent as a module.

Usage:
    python -m wikipedia_agent --provider ollama "What is gravity?"
"""

from wikipedia_agent.cli import main_v2

if __name__ == "__main__":
    main_v2()
