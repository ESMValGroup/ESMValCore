"""Collection of jinja2 templates to render html output."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

file_loader = FileSystemLoader(Path(__file__).parent)
environment = Environment(loader=file_loader, autoescape=True)
get_template = environment.get_template

__all__ = [
    'get_template',
]
