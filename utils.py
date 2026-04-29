# utils.py - UTF-8 console helper
import sys
import io

def setup_utf8():
    """Configure stdout/stderr for UTF-8 output on Windows."""
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')