"""
Reading queries back as English.
"""

from __future__ import annotations

from krrood.entity_query_language import factories
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.symbol_graph.symbol_graph import Symbol

from cramera_bridge.questions.verbalization import (
    DOCUMENTATION_SITE,
    DocumentedPackage,
    PublishedDocumentation,
    QueryVerbalizer,
)
from .dataset.entities import ShelfItem

# %% wording


def test_a_query_reads_back_as_text_and_escaped_html():
    item = factories.variable(ShelfItem, [ShelfItem(name="anvil")])
    query = factories.an(factories.entity(item).where(item.shelf == "<b>top</b>"))
    verbalization = QueryVerbalizer().verbalize(query)
    assert ShelfItem.__name__ in verbalization.text
    assert "<b>" not in verbalization.html
    assert "&lt;b&gt;" in verbalization.html


# %% links


def test_a_class_of_a_documented_package_links_to_its_page():
    reference = SourceReference(owner_type=Symbol)
    link = PublishedDocumentation().resolve(reference)
    assert link.startswith(f"{DOCUMENTATION_SITE}/{DocumentedPackage.KRROOD}")


def test_a_class_of_any_other_package_has_no_link():
    assert (
        PublishedDocumentation().resolve(SourceReference(owner_type=ShelfItem)) is None
    )
