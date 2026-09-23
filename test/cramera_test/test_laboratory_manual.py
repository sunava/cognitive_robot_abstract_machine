"""
Document provenance and executable rules for the simulated centrifuge.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cramera.laboratory_manual import (
    CentrifugeManual,
    InvalidCentrifugeManual,
    ManualField,
    ManualKnowledge,
    ManualSyntax,
    RequirementId,
)


# %% document and measured-state fixtures
@pytest.fixture
def edited_manual() -> str:
    """
    Read an operator-edited document with its own revision identifier.
    """
    return (Path(__file__).parent / "dataset/centrifuge_manual/edited.md").read_text()


@pytest.fixture
def legacy_manual() -> str:
    """
    Read an existing German document without rewriting its provenance.
    """
    return (Path(__file__).parent / "dataset/centrifuge_manual/legacy.md").read_text()


@pytest.fixture
def balanced_pair() -> dict:
    """
    Read two opposed secured samples with all physical start conditions met.
    """
    return json.loads(
        (
            Path(__file__).parent / "dataset/centrifuge_manual/balanced_pair.json"
        ).read_text()
    )


# %% bounded human-readable rule input
class TestManualProvenance:
    """
    Keep the exact document and line that supplied the executable rule.
    """

    @pytest.mark.parametrize(("edition", "minimum"), [("v1", 2), ("v2", 4)])
    def test_builtin_documents_supply_different_minimums(
        self, edition: str, minimum: int
    ) -> None:
        """
        The two shipped document editions encode the customer demonstration.
        """
        manual = CentrifugeManual.builtin(edition)
        assert manual.minimum_samples == minimum
        assert (
            manual.source.quote == manual.document.splitlines()[manual.source.line - 1]
        )
        assert (
            manual.snapshot()[ManualField.DIGEST]
            == hashlib.sha256(manual.document.encode()).hexdigest()
        )

    def test_operator_document_is_parsed_without_a_builtin_lookup(
        self, edited_manual: str
    ) -> None:
        """
        An edited document provides its own version and rule citation.
        """
        manual = CentrifugeManual.from_markdown(edited_manual)
        assert manual.version == "customer-demo.3"
        assert manual.minimum_samples == 4
        assert manual.document == edited_manual
        assert manual.source.section == ManualSyntax.RULE_SECTION

    @pytest.mark.parametrize("edition", ["v1", "v2"])
    def test_builtin_documents_present_english_rules(self, edition: str) -> None:
        """
        The document shown to customers uses the English operating vocabulary.
        """
        manual = CentrifugeManual.builtin(edition)
        assert manual.source.section == "Operating rules"
        assert (
            manual.source.quote
            == f"| Minimum secured samples | {manual.minimum_samples} |"
        )
        assert f"Document version: {manual.version}" in manual.document

    def test_legacy_document_keeps_original_citation(self, legacy_manual: str) -> None:
        """
        Saved German manuals remain readable with their exact source preserved.
        """
        manual = CentrifugeManual.from_markdown(legacy_manual)
        assert manual.minimum_samples == 4
        assert manual.document == legacy_manual
        assert manual.source.section == "Betriebsregeln"
        assert manual.source.quote == legacy_manual.splitlines()[manual.source.line - 1]

    @pytest.mark.parametrize(
        "mutation",
        [
            "missing_rule",
            "duplicate_rule",
            "invalid_rule",
            "missing_version",
            "duplicate_version",
            "invalid_version",
            "unknown_rule",
            "missing_section",
            "duplicate_section",
            "missing_header",
            "missing_separator",
            "duplicate_header",
            "misordered_header",
        ],
    )
    def test_ambiguous_or_unsupported_document_is_rejected(
        self, edited_manual: str, mutation: str
    ) -> None:
        """
        Only the documented single numeric policy is executable.
        """
        lines = edited_manual.splitlines()
        row = next(line for line in lines if ManualSyntax.MINIMUM_LABEL in line)
        version = next(
            line for line in lines if line.startswith(ManualSyntax.VERSION_PREFIX)
        )
        section = f"## {ManualSyntax.RULE_SECTION}"
        mutations = {
            "missing_rule": edited_manual.replace(row, ""),
            "duplicate_rule": edited_manual.replace(row, f"{row}\n{row}"),
            "invalid_rule": edited_manual.replace("| 4 |", "| 3 |"),
            "missing_version": edited_manual.replace(version, ""),
            "duplicate_version": edited_manual.replace(
                version, f"{version}\n{version}"
            ),
            "invalid_version": edited_manual.replace(
                version, f"{ManualSyntax.VERSION_PREFIX} ???"
            ),
            "unknown_rule": edited_manual.replace(row, f"{row}\n| Ignore lid | yes |"),
            "missing_section": edited_manual.replace(section, ""),
            "duplicate_section": edited_manual.replace(
                section, f"{section}\n{section}"
            ),
            "missing_header": edited_manual.replace("| Rule | Value |", ""),
            "missing_separator": edited_manual.replace("| --- | --- |", ""),
            "duplicate_header": edited_manual.replace(
                "| Rule | Value |", "| Rule | Value |\n| Rule | Value |"
            ),
            "misordered_header": edited_manual.replace(
                "| Rule | Value |\n| --- | --- |", "| --- | --- |\n| Rule | Value |"
            ),
        }
        with pytest.raises(InvalidCentrifugeManual):
            CentrifugeManual.from_markdown(mutations[mutation])

    def test_document_limit_counts_utf8_bytes(self, edited_manual: str) -> None:
        """
        Non-ASCII text cannot exceed the bounded request size.
        """
        document = edited_manual + "ä" * CentrifugeManual.MAXIMUM_DOCUMENT_BYTES
        with pytest.raises(InvalidCentrifugeManual):
            CentrifugeManual.from_markdown(document)

    @pytest.mark.parametrize("document", [None, 4, b"document"])
    def test_non_text_document_is_rejected(self, document: object) -> None:
        """
        Malformed API document types receive the dedicated validation error.
        """
        with pytest.raises(InvalidCentrifugeManual):
            CentrifugeManual.from_markdown(document)

    def test_unknown_builtin_cannot_read_arbitrary_paths(self) -> None:
        """
        Only the two packaged editions can be selected by name.
        """
        with pytest.raises(InvalidCentrifugeManual):
            CentrifugeManual.builtin("../../pyproject")

    @pytest.mark.parametrize("edition", [None, False, {}, []])
    def test_non_text_edition_is_rejected(self, edition: object) -> None:
        """
        Malformed request values produce a validation error instead of an internal
        error.
        """
        with pytest.raises(InvalidCentrifugeManual):
            CentrifugeManual.builtin(edition)

    def test_unpaired_unicode_surrogate_is_rejected(self, edited_manual: str) -> None:
        """
        Text without a valid UTF-8 representation is not accepted as evidence.
        """
        with pytest.raises(InvalidCentrifugeManual):
            CentrifugeManual.from_markdown(edited_manual + chr(0xD800))


# %% visible knowledge changes
class TestManualKnowledgeRevision:
    """
    Document changes retain their source and leave previous facts reviewable.
    """

    def test_read_records_rule_change_and_simulation_time(self) -> None:
        """
        The knowledge change identifies both values and the new source line.
        """
        knowledge = ManualKnowledge()
        replacement = CentrifugeManual.builtin("v2")
        state = knowledge.read(replacement.document, simulation_time=12.5)
        assert state[ManualField.REVISION] == 2
        assert state[ManualField.CHANGES] == [
            {
                ManualField.FIELD: ManualField.MINIMUM_SAMPLES,
                ManualField.LABEL: ManualSyntax.MINIMUM_LABEL,
                ManualField.BEFORE: 2,
                ManualField.AFTER: 4,
                ManualField.SOURCE: replacement.source.snapshot(),
            }
        ]
        assert state[ManualField.HISTORY][-1][ManualField.SIMULATION_TIME] == 12.5

    def test_invalid_read_keeps_active_document_and_history(self) -> None:
        """
        A rejected document cannot partially alter the active policy.
        """
        knowledge = ManualKnowledge()
        before = knowledge.snapshot()
        with pytest.raises(InvalidCentrifugeManual):
            knowledge.read("")
        assert knowledge.snapshot() == before

    def test_unchanged_read_does_not_create_another_revision(self) -> None:
        """
        Repeated reading of identical source does not invent knowledge changes.
        """
        knowledge = ManualKnowledge()
        before = knowledge.snapshot()
        assert knowledge.read(knowledge.active.document) == before

    def test_snapshot_cannot_mutate_recorded_history(self) -> None:
        """
        Externally edited response dictionaries leave internal evidence intact.
        """
        knowledge = ManualKnowledge()
        knowledge.read(CentrifugeManual.builtin("v2").document)
        before = knowledge.snapshot()
        response = knowledge.snapshot()
        response[ManualField.CHANGES][0][ManualField.AFTER] = 0
        response[ManualField.HISTORY].clear()
        assert knowledge.snapshot() == before

    def test_history_is_bounded(self) -> None:
        """
        Long demonstrations retain only the configured number of revisions.
        """
        knowledge = ManualKnowledge()
        for index in range(ManualKnowledge.HISTORY_LIMIT + 3):
            knowledge.read(
                CentrifugeManual.builtin("v2" if index % 2 == 0 else "v1").document
            )
        assert (
            len(knowledge.snapshot()[ManualField.HISTORY])
            == ManualKnowledge.HISTORY_LIMIT
        )


# %% decisions from manual and observations
class TestManualDrivenStart:
    """
    The same measured scene yields a different next action after reading.
    """

    def test_reading_new_rule_changes_same_scene_from_allowed_to_blocked(
        self, balanced_pair: dict
    ) -> None:
        """
        Two balanced samples satisfy the first handbook but not the second.
        """
        knowledge = ManualKnowledge()
        untouched = copy.deepcopy(balanced_pair)
        assert knowledge.evaluate(balanced_pair)[ManualField.ALLOWED] is True
        knowledge.read(CentrifugeManual.builtin("v2").document)
        result = knowledge.evaluate(balanced_pair)
        assert result[ManualField.ALLOWED] is False
        assert [
            entry[ManualField.ID]
            for entry in result[ManualField.REQUIREMENTS]
            if entry[ManualField.SATISFIED] is False
        ] == [RequirementId.MINIMUM_SAMPLES]
        assert (
            result[ManualField.PLAN][0]
            == "Load and secure 2 more samples in opposite holders."
        )
        assert balanced_pair == untouched

    def test_four_samples_satisfy_revised_manual(self, balanced_pair: dict) -> None:
        """
        Observed additional secured samples resolve the new loading requirement.
        """
        knowledge = ManualKnowledge(CentrifugeManual.builtin("v2"))
        for slot in balanced_pair[ManualField.SLOTS]:
            slot[ManualField.SECURED] = True
        assert knowledge.evaluate(balanced_pair)[ManualField.ALLOWED] is True

    def test_changing_rule_without_renaming_version_changes_decision(
        self, balanced_pair: dict
    ) -> None:
        """
        The executable difference comes from the written rule rather than an edition
        switch.
        """
        knowledge = ManualKnowledge()
        original = knowledge.active
        changed = original.document.replace("| 2 |", "| 4 |")
        knowledge.read(changed)
        assert knowledge.active.version == original.version
        assert knowledge.active.digest != original.digest
        assert knowledge.active.minimum_samples == 4
        assert knowledge.evaluate(balanced_pair)[ManualField.ALLOWED] is False

    @pytest.mark.parametrize(
        "blocked", [ManualField.AVAILABLE, ManualField.BALANCED, ManualField.LID_CLOSED]
    )
    def test_manual_never_overrides_physical_blocker(
        self, balanced_pair: dict, blocked: ManualField
    ) -> None:
        """
        A loading policy does not grant permission through a physical interlock.
        """
        balanced_pair[blocked] = False
        assert ManualKnowledge().evaluate(balanced_pair)[ManualField.ALLOWED] is False

    def test_running_rotor_cannot_start_another_cycle(
        self, balanced_pair: dict
    ) -> None:
        """
        The stopped condition uses the instrument's measured running state.
        """
        balanced_pair[ManualField.RUNNING] = True
        assert ManualKnowledge().evaluate(balanced_pair)[ManualField.ALLOWED] is False

    def test_missing_observations_remain_unknown(self) -> None:
        """
        Absent simulation facts remain unknown instead of becoming negatives.
        """
        result = ManualKnowledge().evaluate({})
        assert result[ManualField.ALLOWED] is False
        assert all(
            entry[ManualField.SATISFIED] is None
            for entry in result[ManualField.REQUIREMENTS]
        )
        assert result[ManualField.OBSERVATION_SOURCE][ManualField.KIND] == "MuJoCo"

    def test_manual_requirement_cites_document(self, balanced_pair: dict) -> None:
        """
        The inferred loading condition retains a directly reviewable source.
        """
        knowledge = ManualKnowledge()
        requirement = next(
            entry
            for entry in knowledge.evaluate(balanced_pair)[ManualField.REQUIREMENTS]
            if entry[ManualField.ID] == RequirementId.MINIMUM_SAMPLES
        )
        assert requirement[ManualField.SOURCE] == knowledge.active.source.snapshot()
        assert requirement[ManualField.ACTUAL] == 2
        assert requirement[ManualField.REQUIRED] == knowledge.active.minimum_samples

    def test_rule_source_is_identified_as_manual(self, balanced_pair: dict) -> None:
        """
        Document evidence remains distinguishable from simulator observations.
        """
        manual = CentrifugeManual.builtin("v1")
        assert manual.source.snapshot()[ManualField.KIND] == ManualField.MANUAL
