import json
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[2] / "pycram/demos/thesis"
sys.path.insert(0, str(BASE_DIR))

import extract_action_cases as extract_mod
import download_wikihow_articles as download_mod
import llm_case_generator as llm_mod
import score_template_fit as score_mod
import scrape_wikihow_actions as scrape_mod
from wikihow_eval import models as models_mod
from wikihow_eval import ontology as ontology_mod
from wikihow_eval import templates as templates_mod


def test_extract_case_parses_title_and_tool_hint():
    article = models_mod.WikiHowArticle(
        title="How to Cut Carrots",
        categories=["Food and Entertaining", "Vegetables"],
        steps=["Use a chef's knife to cut the carrots into coins."],
    )

    case = extract_mod.extract_case(article)

    assert case.verb == "cut"
    assert case.object_text == "carrots"
    assert case.tool_hint == "knife"
    assert case.domain_hint == "food_preparation"


def test_extract_case_handles_prefixed_title_variants():
    article = models_mod.WikiHowArticle(
        title="Titanium Dioxide Explained: How to Mix and Use It",
        categories=["Crafts"],
        steps=["Use a spoon to mix the powder into liquid."],
    )

    case = extract_mod.extract_case(article)

    assert case.verb == "mix"
    assert case.object_text == "titanium dioxide"


def test_mapping_marks_cut_hair_as_grooming_body_part():
    action_case = models_mod.ActionCase(
        title="How to Cut Hair",
        verb="cut",
        action_word="cut",
        object_text="hair",
        tool_hint="scissors",
        domain_hint="grooming",
        categories=["Personal Care and Style", "Hair Care"],
        steps=["Use scissors to trim the ends evenly."],
    )

    ontology_case = ontology_mod.map_case_to_ontology(action_case)

    assert ontology_case.object_class == "BodyPart"
    assert ontology_case.tool_class == "CuttingTool"
    assert ontology_case.domain == "grooming"


def test_scoring_distinguishes_full_partial_and_out_of_scope():
    full_case = models_mod.OntologyCase(
        title="How to Cut Bread",
        verb="cut",
        template_candidates=["cutting"],
        object_text="bread",
        object_class="FoodItem",
        tool_text="bread knife",
        tool_class="CuttingTool",
        domain="food_preparation",
        material_class="FoodMaterial",
        functional_tags=["cuttable", "requires_sharp_tool", "separable"],
        categories=[],
        steps=[],
    )
    partial_case = models_mod.OntologyCase(
        title="How to Mix Cement",
        verb="mix",
        template_candidates=["mixing"],
        object_text="cement",
        object_class="ConstructionMaterial",
        tool_text="spoon",
        tool_class="MixingTool",
        domain="construction",
        material_class="ConstructionMaterial",
        functional_tags=["mixable"],
        categories=[],
        steps=[],
    )
    out_case = models_mod.OntologyCase(
        title="How to Cut Hair",
        verb="cut",
        template_candidates=["cutting"],
        object_text="hair",
        object_class="BodyPart",
        tool_text="scissors",
        tool_class="CuttingTool",
        domain="grooming",
        material_class="LivingTissue",
        functional_tags=["animate_part", "cuttable", "requires_sharp_tool"],
        categories=[],
        steps=[],
    )

    full_result = templates_mod.score_case(full_case)[0]
    partial_result = templates_mod.score_case(partial_case)[0]
    out_result = templates_mod.score_case(out_case)[0]

    assert full_result.fit == "full_fit"
    assert partial_result.fit == "partial_fit"
    assert out_result.fit == "out_of_scope"


def test_scraper_deduplicates_and_filters_large_batches():
    articles = [
        models_mod.WikiHowArticle(
            title="How to Cut Carrots",
            categories=[],
            steps=[],
            url="https://example/cut-carrots",
        ),
        models_mod.WikiHowArticle(
            title="How to Cut Carrots",
            categories=[],
            steps=[],
            url="https://example/cut-carrots",
        ),
        models_mod.WikiHowArticle(
            title="How to Mix Batter",
            categories=[],
            steps=[],
            url="https://example/mix-batter",
        ),
        models_mod.WikiHowArticle(
            title="How to Knit Socks",
            categories=[],
            steps=[],
            url="https://example/knit-socks",
        ),
    ]

    deduped = scrape_mod.deduplicate_articles(articles, dedupe_by="url")
    filtered = scrape_mod.filter_articles(deduped, ["cut", "mix"])

    assert len(deduped) == 3
    assert [article.title for article in filtered] == [
        "How to Cut Carrots",
        "How to Mix Batter",
    ]


def test_scraper_matches_prefixed_how_to_titles():
    articles = [
        models_mod.WikiHowArticle(
            title="Titanium Dioxide Explained: How to Mix and Use It",
            categories=[],
            steps=[],
            url="https://example/mix",
        ),
        models_mod.WikiHowArticle(
            title="Kitchen Notes: How to Chop Herbs",
            categories=[],
            steps=[],
            url="https://example/chop",
        ),
    ]

    filtered = scrape_mod.filter_articles(articles, ["mix", "chop"])

    assert [article.title for article in filtered] == [
        "Titanium Dioxide Explained: How to Mix and Use It",
        "Kitchen Notes: How to Chop Herbs",
    ]


def test_kitchen_filter_keeps_food_articles_and_drops_grooming():
    articles = [
        models_mod.WikiHowArticle(
            title="How to Cut Carrots",
            categories=["Food and Entertaining"],
            steps=["Use a knife on a cutting board."],
            url="https://example/cut-carrots",
        ),
        models_mod.WikiHowArticle(
            title="How to Cut Hair",
            categories=["Personal Care and Style"],
            steps=["Use scissors to trim hair."],
            url="https://example/cut-hair",
        ),
    ]

    filtered = scrape_mod.filter_kitchen_articles(articles)

    assert [article.title for article in filtered] == ["How to Cut Carrots"]


def test_kitchen_filter_drops_health_cut_articles():
    articles = [
        models_mod.WikiHowArticle(
            title="4 Ways to Heal Cuts Quickly (Using Easy, Natural Items) - wikiHow",
            categories=["Health", "First Aid and Emergency Health Care"],
            steps=["Clean the wound and apply a bandage."],
            url="https://example/heal-cuts",
        ),
        models_mod.WikiHowArticle(
            title="How to Slice Bread",
            categories=["Food and Entertaining"],
            steps=["Use a bread knife on a cutting board."],
            url="https://example/slice-bread",
        ),
    ]

    filtered = scrape_mod.filter_kitchen_articles(articles)

    assert [article.title for article in filtered] == ["How to Slice Bread"]


def test_llm_response_parser_creates_action_cases():
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "cases": [
                                {
                                    "title": "How to Cut a Banana",
                                    "verb": "cut",
                                    "object_text": "banana",
                                    "tool_hint": "knife",
                                    "domain_hint": "food_preparation",
                                    "categories": ["Food and Entertaining"],
                                    "steps": [
                                        "Place the banana on a cutting board.",
                                        "Use a knife to cut the banana into slices.",
                                    ],
                                    "expected_scope": "in_scope",
                                    "rationale": "Food item with a cutting tool in a kitchen context.",
                                },
                                {
                                    "title": "How to Cut Hair",
                                    "verb": "cut",
                                    "object_text": "hair",
                                    "tool_hint": "scissors",
                                    "domain_hint": "grooming",
                                    "categories": ["Personal Care and Style"],
                                    "steps": ["Trim the hair with scissors."],
                                    "expected_scope": "out_of_scope",
                                    "rationale": "Grooming domain, not food preparation.",
                                },
                            ]
                        }
                    )
                }
            }
        ]
    }

    cases = llm_mod.parse_cases_from_response(response_payload)

    assert [case.title for case in cases] == ["How to Cut a Banana", "How to Cut Hair"]
    assert cases[0].source == "llm"
    assert cases[0].metadata["expected_scope"] == "in_scope"
    assert cases[1].domain_hint == "grooming"


def test_llm_prompt_mentions_verbs_domains_and_templates():
    prompt = llm_mod.build_prompt(
        verbs=["cut", "mix"],
        domains=["food_preparation", "grooming"],
        cases_per_verb=6,
        template_names=["cutting", "mixing"],
    )

    assert "cut, mix" in prompt
    assert "food_preparation, grooming" in prompt
    assert "cutting, mixing" in prompt


def test_llm_chunked_splits_verb_batches():
    assert llm_mod.chunked(["cut", "mix", "pour"], 2) == [
        ["cut", "mix"],
        ["pour"],
    ]


def test_coverage_report_counts_out_of_scope_clusters():
    cases = [
        models_mod.OntologyCase(
            title="How to Cut Hair",
            verb="cut",
            template_candidates=["cutting"],
            object_text="hair",
            object_class="BodyPart",
            tool_text="scissors",
            tool_class="CuttingTool",
            domain="grooming",
            material_class="LivingTissue",
            functional_tags=["animate_part", "cuttable"],
            categories=[],
            steps=[],
        ),
        models_mod.OntologyCase(
            title="How to Cut Bread",
            verb="cut",
            template_candidates=["cutting"],
            object_text="bread",
            object_class="FoodItem",
            tool_text="knife",
            tool_class="CuttingTool",
            domain="food_preparation",
            material_class="FoodMaterial",
            functional_tags=["cuttable"],
            categories=[],
            steps=[],
        ),
    ]

    results = []
    for case in cases:
        for result in templates_mod.score_case(case):
            entry = result.to_dict()
            entry["fit_case_verb"] = case.verb
            entry["fit_case_object"] = case.object_text
            results.append(entry)

    report = score_mod.build_coverage_report(cases, results)

    assert report["article_count"] == 2
    assert report["template_coverage"]["cutting"]["out_of_scope"] == 1
    assert report["top_out_of_scope_domains"][0] == ("grooming", 1)


def test_download_parser_extracts_article_urls_from_search_html():
    html = """
    <html><body>
      <a class="result_link" href="/Cut-Outside-Container">outside</a>
      <div id="searchresults_list">
        <a class="result_link" href="/Cut-Carrots">Cut carrots</a>
        <a class="result_link" href="https://www.wikihow.com/Mix-Batter">Mix batter</a>
        <a href="/Cut-No-Class">wrong anchor</a>
        <a class="result_link" href="/Quizzes">quiz</a>
        <a class="result_link" href="/Main-Page">main</a>
      </div>
      <a class="result_link" href="/Cut-After-Container">after</a>
    </body></html>
    """

    urls = download_mod.extract_article_urls(html, query_term="cut")

    assert urls == ["https://www.wikihow.com/Cut-Carrots"]


def test_download_parser_prefers_json_ld_howto_blocks():
    html = """
    <html><head>
      <script type="application/ld+json">
      {
        "@type": "HowTo",
        "name": "How to Cut Carrots",
        "step": [
          {"@type": "HowToStep", "text": "Wash the carrots."},
          {"@type": "HowToStep", "text": "Use a knife to slice them."}
        ]
      }
      </script>
    </head><body></body></html>
    """

    article = download_mod.parse_article_html(
        html, "https://www.wikihow.com/Cut-Carrots", "cut"
    )

    assert article.title == "How to Cut Carrots"
    assert article.steps == ["Wash the carrots.", "Use a knife to slice them."]
    assert article.source_query == "cut"
