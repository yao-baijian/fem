"""
Logger stub for backward compatibility.
This is a temporary stub to support drawer.py until it's refactored.
"""

def INFO(*args):
    """Log info message."""
    print("[INFO]", *args)

def WARNING(*args):
    """Log warning message."""
    print("[WARNING]", *args)

def ERROR(*args):
    """Log error message."""
    print("[ERROR]", *args)

def DEBUG(*args):
    """Log debug message."""
    print("[DEBUG]", *args)
