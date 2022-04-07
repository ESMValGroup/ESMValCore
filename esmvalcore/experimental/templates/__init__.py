"""Collection of jinja2 templates to render html output."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = str(Path(__file__).parent)
file_loader = FileSystemLoader(TEMPLATE_DIR)
environment = Environment(loader=file_loader, autoescape=True)
get_template = environment.get_template

__all__ = [
    'get_template',
]
