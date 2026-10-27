"""
Protocol documents stored as JSON files.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Optional, Type, TypeVar

from cramera_protocol.document import JsonDocument

DocumentType = TypeVar("DocumentType", bound=JsonDocument)
"""
The class of a document read from a file.
"""

JSON_INDENT = 1
"""
The indentation of written files, which keeps them readable in a diff.
"""

# %% files


@dataclass(frozen=True)
class DocumentFile:
    """
    One JSON file holding a protocol document.
    """

    path: Path
    """
    The file.
    """

    def write(self, document: JsonDocument) -> None:
        """
        Write the document so a reader sees either the old or the new file, never a
        partly written one.

        :param document: The document to write.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document.to_json(), stream, indent=JSON_INDENT)
        os.replace(temporary, self.path)

    def read(self, document_class: Type[DocumentType]) -> DocumentType:
        """
        :param document_class: The class of the stored document.
        :return: The document.
        :raises FileNotFoundError: When the file does not exist.
        :raises ProtocolError: When the file does not hold such a document.
        """
        return document_class.from_json(
            json.loads(self.path.read_text(encoding="utf-8"))
        )

    def read_if_present(
        self, document_class: Type[DocumentType]
    ) -> Optional[DocumentType]:
        """
        :param document_class: The class of the stored document.
        :return: The document, or None when the file does not exist.
        """
        if not self.path.is_file():
            return None
        return self.read(document_class)
