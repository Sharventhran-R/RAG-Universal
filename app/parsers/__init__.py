"""Parser package.

Import each concrete parser module here so its ``register_parser(...)`` call
runs at startup. Remaining Phase 1 wiring: ``text``, ``html``.
"""

from app.parsers import docx, pdf, pptx, spreadsheet  # noqa: F401

