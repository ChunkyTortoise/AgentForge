"""
Safe File System Tools for Agents.
Allows agents to read the codebase but restricts them to the current working directory.
"""
import os
from typing import List, Optional
from langchain_core.tools import tool
from utils.logger import get_logger

logger = get_logger(__name__)

# Security: Restrict to current directory and below
BASE_DIR = os.path.abspath(".")

def _is_safe_path(path: str) -> bool:
    """Ensure path is within the project root."""
    abs_path = os.path.abspath(path)
    return abs_path.startswith(BASE_DIR)

@tool
def list_files(directory: str = ".") -> str:
    """
    List files in a directory. 
    Use this to explore the codebase structure.
    Args:
        directory: Relative path to list (default: ".")
    """
    try:
        if not _is_safe_path(directory):
            return "Error: Access denied. Path outside project root."
        
        items = os.listdir(directory)
        # Filter out hidden files and venv
        items = [i for i in items if not i.startswith(".") and i != "__pycache__" and i != "venv"]
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {e}"

@tool
def read_file(file_path: str) -> str:
    """
    Read the content of a file.
    Use this to examine code, config files, or documentation.
    Args:
        file_path: Relative path to the file.
    """
    try:
        if not _is_safe_path(file_path):
            return "Error: Access denied. Path outside project root."
        
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Truncate if too large to prevent context overflow
        if len(content) > 20000:
            return content[:20000] + "\n...[TRUNCATED]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"
