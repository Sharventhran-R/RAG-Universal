"""Parser package.

Import each concrete parser module here so its ``register_parser(...)`` call
runs at startup.
"""

from app.parsers import docx, html, pdf, pptx, spreadsheet, text  # noqa: F401

