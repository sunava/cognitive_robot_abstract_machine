"""
Natural-language verbalization for EQL expression trees.

Plain-text API (backward-compatible)::

    from krrood.entity_query_language.verbalization.verbalizer import verbalize_expression
    text = verbalize_expression(expression)

Coloured / formatted API::

    from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
    text = VerbalizationPipeline.markdown(hierarchical=True).verbalize(expression)
"""
