"""Trace document-supplied loading rules to visible centrifuge decisions."""

from __future__ import annotations

import copy
import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.resources import files

from typing_extensions import Any, ClassVar


# %% document and response vocabulary
class ManualSyntax(StrEnum):
    """Recognized elements of the deliberately bounded Markdown document format."""

    VERSION_PREFIX = "Dokumentversion:"
    """Line prefix identifying the revision of the supplied document."""
    RULE_SECTION = "Betriebsregeln"
    """Level-two section containing the executable policy table."""
    MINIMUM_LABEL = "Mindestzahl gesicherter Proben"
    """Supported row naming the minimum number of secured vessels."""
    HEADER_RULE = "Regel"
    """First column heading of the policy table."""
    HEADER_VALUE = "Wert"
    """Second column heading of the policy table."""
    RESOURCE_DIRECTORY = "resources/centrifuge_manuals"
    """Packaged location of the demonstration handbooks."""
    PACKAGE = "cramera"
    """Python package owning the handbook resources."""


class ManualEdition(StrEnum):
    """Packaged demonstration editions accepted by name."""

    FIRST = "v1"
    """Edition permitting one opposed pair."""
    SECOND = "v2"
    """Edition requiring both opposed pairs."""


class ManualField(StrEnum):
    """Keys of the document knowledge and measured instrument interfaces."""

    TITLE = "title"
    """Human-readable document heading."""
    VERSION = "version"
    """Document-authored revision label."""
    DOCUMENT = "document"
    """Exact Markdown read by the rule interpreter."""
    DIGEST = "digest"
    """SHA-256 digest of the exact UTF-8 document."""
    MINIMUM_SAMPLES = "minimumSamples"
    """Minimum number of secured samples required by the document."""
    SOURCE = "source"
    """Traceable origin of a rule or observation."""
    SECTION = "section"
    """Document section containing a cited rule."""
    LINE = "line"
    """One-based line of the original document."""
    QUOTE = "quote"
    """Unmodified document line supplying the rule."""
    MANUAL = "manual"
    """Active interpreted handbook and its exact source."""
    REVISION = "revision"
    """Monotonically increasing knowledge revision."""
    CHANGES = "changes"
    """Executable rule changes from the latest document read."""
    HISTORY = "history"
    """Bounded history of accepted document revisions."""
    EVALUATION = "evaluation"
    """Current start requirements derived from manual and observations."""
    FIELD = "field"
    """Rule or observed fact addressed by an entry."""
    LABEL = "label"
    """Explanation suitable for the operator interface."""
    BEFORE = "before"
    """Executable rule value before a document change."""
    AFTER = "after"
    """Executable rule value after a document change."""
    SIMULATION_TIME = "simulationTime"
    """Optional simulation time at the moment of reading."""
    ALLOWED = "allowed"
    """Whether every evaluated start requirement is known to hold."""
    REQUIREMENTS = "requirements"
    """Individual requirements with observations and provenance."""
    PLAN = "plan"
    """Suggested steps for satisfying the evaluated requirements."""
    SATISFIED = "satisfied"
    """True, false, or unknown according to the available observations."""
    ACTUAL = "actual"
    """Currently observed value, or None if unknown."""
    REQUIRED = "required"
    """Condition needed for the evaluated requirement."""
    ID = "id"
    """Stable requirement identifier."""
    OBSERVATION_SOURCE = "observationSource"
    """Provenance shared by all measured values in an evaluation."""
    KIND = "kind"
    """Category of evidence supplying an observation."""
    AVAILABLE = "available"
    """Whether the instrument exists in the observed simulation."""
    BALANCED = "balanced"
    """Whether opposing secured samples have balanced measured mass."""
    LID_CLOSED = "lidClosed"
    """Whether the physical lid is observed fully closed."""
    RUNNING = "running"
    """Whether the instrument is observed moving or driving its rotor."""
    SLOTS = "slots"
    """Observed rotor holder records."""
    SECURED = "secured"
    """Whether a holder's fixture is engaged."""


class RequirementId(StrEnum):
    """Stable identities of evaluated start requirements."""

    AVAILABLE = "available"
    """The instrument is present."""
    MINIMUM_SAMPLES = "minimumSamples"
    """The document-supplied loading minimum is met."""
    BALANCED = "balanced"
    """The loaded rotor is balanced."""
    LID_CLOSED = "lidClosed"
    """The physical cover is closed."""
    STOPPED = "stopped"
    """The previous rotor motion has stopped."""


class ObservationSource(StrEnum):
    """Visible disclosure of where simulated observations originate."""

    KIND = "MuJoCo"
    """Authoritative simulator supplying the observations."""
    LABEL = "Direkter Simulationszustand (keine visuelle Erkennung)"
    """Explanation distinguishing state access from visual recognition."""


@dataclass
class InvalidCentrifugeManual(ValueError):
    """A document cannot supply an unambiguous supported loading rule."""

    reason: str
    """Problem to correct before submitting the document again."""

    def __str__(self) -> str:
        """Expose the validation explanation to the operator."""
        return self.reason


@dataclass(frozen=True)
class ManualSource:
    """Exact handbook location of an executable operating rule."""

    section: str
    """Section heading containing the rule."""
    line: int
    """One-based line number in the unmodified document."""
    quote: str
    """Exact line parsed to derive the executable rule."""

    def snapshot(self) -> dict[str, Any]:
        """Expose a directly reviewable document citation."""
        return {
            ManualField.KIND: ManualField.MANUAL,
            ManualField.SECTION: self.section,
            ManualField.LINE: self.line,
            ManualField.QUOTE: self.quote,
        }


# %% validated handbook input
@dataclass(frozen=True)
class CentrifugeManual:
    """One validated document and the loading rule explicitly written in it."""

    title: str
    """Human-readable first-level document heading."""
    version: str
    """Revision identifier supplied by the document author."""
    document: str
    """Unmodified accepted Markdown source."""
    digest: str
    """SHA-256 digest of the exact accepted source."""
    minimum_samples: int
    """Number of secured samples required before starting."""
    source: ManualSource
    """Exact location supplying the minimum sample count."""
    MAXIMUM_DOCUMENT_BYTES: ClassVar[int] = 8192
    """Maximum UTF-8 document size accepted for a manual read."""
    ALLOWED_MINIMUMS: ClassVar[tuple[int, ...]] = (2, 4)
    """Supported minimum counts for the four-holder simulated rotor."""
    VERSION_PATTERN: ClassVar[str] = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
    """Bounded document version grammar."""
    SURROGATE_PATTERN: ClassVar[str] = r"[\ud800-\udfff]"
    """Unicode surrogate code points that cannot be encoded independently."""

    @classmethod
    def builtin(cls, edition: str) -> CentrifugeManual:
        """Read one of the two packaged demonstration handbooks."""
        if not isinstance(edition, str) or edition not in tuple(ManualEdition):
            raise InvalidCentrifugeManual("Unbekannte Handbuchausgabe.")
        document = (
            files(ManualSyntax.PACKAGE)
            .joinpath(ManualSyntax.RESOURCE_DIRECTORY, f"{edition}.md")
            .read_text(encoding="utf-8")
        )
        return cls.from_markdown(document)

    @classmethod
    def from_markdown(cls, document: str) -> CentrifugeManual:
        """Validate the document and interpret its single supported table rule."""
        if (
            not isinstance(document, str)
            or not document
            or re.search(cls.SURROGATE_PATTERN, document)
            or len(document.encode("utf-8")) > cls.MAXIMUM_DOCUMENT_BYTES
        ):
            raise InvalidCentrifugeManual(
                "Das Handbuch muss Text mit höchstens 8192 UTF-8-Bytes enthalten."
            )
        lines = document.splitlines()
        titles = [line[2:].strip() for line in lines if line.startswith("# ")]
        versions = [
            line.removeprefix(ManualSyntax.VERSION_PREFIX).strip()
            for line in lines
            if line.startswith(ManualSyntax.VERSION_PREFIX)
        ]
        if len(titles) != 1 or not titles[0]:
            raise InvalidCentrifugeManual(
                "Genau eine Handbuchüberschrift ist erforderlich."
            )
        if len(versions) != 1 or re.fullmatch(cls.VERSION_PATTERN, versions[0]) is None:
            raise InvalidCentrifugeManual(
                "Genau eine gültige Dokumentversion ist erforderlich."
            )
        section_indices = [
            index
            for index, line in enumerate(lines)
            if line.strip() == f"## {ManualSyntax.RULE_SECTION}"
        ]
        if len(section_indices) != 1:
            raise InvalidCentrifugeManual(
                "Genau ein Abschnitt Betriebsregeln ist erforderlich."
            )
        rows = []
        header_seen = False
        separator_seen = False
        for index in range(section_indices[0] + 1, len(lines)):
            line = lines[index]
            if line.lstrip().startswith("#"):
                break
            if not line.strip():
                continue
            columns = [column.strip() for column in line.strip().strip("|").split("|")]
            if columns == [ManualSyntax.HEADER_RULE, ManualSyntax.HEADER_VALUE]:
                if header_seen or separator_seen or rows:
                    raise InvalidCentrifugeManual(
                        "Die Regeltabelle benötigt genau eine Kopfzeile."
                    )
                header_seen = True
                continue
            if len(columns) == 2 and all(
                re.fullmatch(r":?-{3,}:?", column) for column in columns
            ):
                if not header_seen or separator_seen or rows:
                    raise InvalidCentrifugeManual(
                        "Die Trennzeile muss unmittelbar auf die Tabellenkopfzeile folgen."
                    )
                separator_seen = True
                continue
            if (
                not header_seen
                or not separator_seen
                or len(columns) != 2
                or columns[0] != ManualSyntax.MINIMUM_LABEL
                or columns[1] not in tuple(str(value) for value in cls.ALLOWED_MINIMUMS)
            ):
                raise InvalidCentrifugeManual(
                    "Betriebsregeln unterstützt nur die Mindestzahl gesicherter Proben mit Wert 2 oder 4."
                )
            rows.append((index, columns[1]))
        if len(rows) != 1:
            raise InvalidCentrifugeManual(
                "Genau eine Regel Mindestzahl gesicherter Proben ist erforderlich."
            )
        index, value = rows[0]
        return cls(
            titles[0],
            versions[0],
            document,
            hashlib.sha256(document.encode("utf-8")).hexdigest(),
            int(value),
            ManualSource(ManualSyntax.RULE_SECTION, index + 1, lines[index]),
        )

    def snapshot(self) -> dict[str, Any]:
        """Expose the source document beside its interpreted loading policy."""
        return {
            ManualField.TITLE: self.title,
            ManualField.VERSION: self.version,
            ManualField.DOCUMENT: self.document,
            ManualField.DIGEST: self.digest,
            ManualField.MINIMUM_SAMPLES: self.minimum_samples,
            ManualField.SOURCE: self.source.snapshot(),
        }


# %% observations and start requirements
@dataclass(frozen=True)
class InstrumentObservation:
    """Known physical facts from one simulator instrument snapshot."""

    available: bool | None
    """Observed instrument availability, or unknown."""
    secured_samples: int | None
    """Observed number of engaged fixtures, or unknown."""
    balanced: bool | None
    """Observed balance result, or unknown."""
    lid_closed: bool | None
    """Observed lid closure, or unknown."""
    stopped: bool | None
    """Observed rotor standstill, or unknown."""

    @staticmethod
    def boolean(snapshot: dict, key: ManualField) -> bool | None:
        """Keep missing or malformed physical facts explicitly unknown."""
        value = snapshot.get(key)
        return value if type(value) is bool else None

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> InstrumentObservation:
        """Extract only explicitly reported facts from the simulator response."""
        slots = snapshot.get(ManualField.SLOTS)
        samples = None
        if isinstance(slots, list) and all(
            isinstance(slot, dict) and type(slot.get(ManualField.SECURED)) is bool
            for slot in slots
        ):
            samples = sum(slot[ManualField.SECURED] for slot in slots)
        running = cls.boolean(snapshot, ManualField.RUNNING)
        return cls(
            cls.boolean(snapshot, ManualField.AVAILABLE),
            samples,
            cls.boolean(snapshot, ManualField.BALANCED),
            cls.boolean(snapshot, ManualField.LID_CLOSED),
            None if running is None else not running,
        )

    @staticmethod
    def source() -> dict[str, str]:
        """Identify direct simulator access as the origin of observed facts."""
        return {
            ManualField.KIND: ObservationSource.KIND,
            ManualField.LABEL: ObservationSource.LABEL,
        }


@dataclass(frozen=True)
class StartRequirement:
    """One explainable comparison between evidence and a required condition."""

    identifier: RequirementId
    """Stable identity of the assessed condition."""
    label: str
    """User-facing name of the condition."""
    actual: bool | int | None
    """Measured value or explicitly unknown."""
    required: bool | int
    """Expected boolean or minimum count."""
    source: dict[str, Any]
    """Document citation or simulator observation provenance."""

    @property
    def satisfied(self) -> bool | None:
        """Evaluate known evidence while retaining unknown conditions."""
        if self.actual is None:
            return None
        if self.identifier == RequirementId.MINIMUM_SAMPLES:
            return self.actual >= self.required
        return self.actual == self.required

    def snapshot(self) -> dict[str, Any]:
        """Expose evidence, threshold, result and provenance together."""
        return {
            ManualField.ID: self.identifier,
            ManualField.LABEL: self.label,
            ManualField.SATISFIED: self.satisfied,
            ManualField.ACTUAL: self.actual,
            ManualField.REQUIRED: self.required,
            ManualField.SOURCE: copy.deepcopy(self.source),
        }


# %% revisioned knowledge and decisions
@dataclass
class ManualKnowledge:
    """Retain accepted handbook revisions and evaluate their loading requirement."""

    active: CentrifugeManual = field(
        default_factory=lambda: CentrifugeManual.builtin(ManualEdition.FIRST)
    )
    """Currently accepted source document and executable loading rule."""
    revision: int = field(init=False, default=1)
    """Knowledge revision incremented for each distinct accepted document."""
    changes: list[dict[str, Any]] = field(init=False, default_factory=list)
    """Rule differences introduced by the latest accepted document."""
    history: deque = field(init=False)
    """Bounded history of source revisions and their executable differences."""
    HISTORY_LIMIT: ClassVar[int] = 20
    """Maximum number of document revisions retained in the visible history."""

    def __post_init__(self) -> None:
        """Record the initial source as the first knowledge revision."""
        self.history = deque(maxlen=self.HISTORY_LIMIT)
        self.record_revision(None)

    def record_revision(self, simulation_time: float | None) -> None:
        """Record accepted document identity and the rule changes it introduced."""
        self.history.append(
            {
                ManualField.REVISION: self.revision,
                ManualField.VERSION: self.active.version,
                ManualField.DIGEST: self.active.digest,
                ManualField.CHANGES: copy.deepcopy(self.changes),
                ManualField.SIMULATION_TIME: simulation_time,
            }
        )

    def read(
        self, document: str, simulation_time: float | None = None
    ) -> dict[str, Any]:
        """Apply a valid document atomically and return its visible knowledge change."""
        replacement = CentrifugeManual.from_markdown(document)
        if replacement.digest == self.active.digest:
            return self.snapshot()
        changes = []
        if replacement.minimum_samples != self.active.minimum_samples:
            changes.append(
                {
                    ManualField.FIELD: ManualField.MINIMUM_SAMPLES,
                    ManualField.LABEL: ManualSyntax.MINIMUM_LABEL,
                    ManualField.BEFORE: self.active.minimum_samples,
                    ManualField.AFTER: replacement.minimum_samples,
                    ManualField.SOURCE: replacement.source.snapshot(),
                }
            )
        self.active = replacement
        self.changes = changes
        self.revision += 1
        self.record_revision(simulation_time)
        return self.snapshot()

    def evaluate(self, instrument_snapshot: dict) -> dict[str, Any]:
        """Derive start requirements and next steps from handbook and simulator facts."""
        observed = InstrumentObservation.from_snapshot(instrument_snapshot)
        requirements = [
            StartRequirement(
                RequirementId.AVAILABLE,
                "Zentrifuge vorhanden",
                observed.available,
                True,
                {**observed.source(), ManualField.FIELD: ManualField.AVAILABLE},
            ),
            StartRequirement(
                RequirementId.MINIMUM_SAMPLES,
                ManualSyntax.MINIMUM_LABEL,
                observed.secured_samples,
                self.active.minimum_samples,
                self.active.source.snapshot(),
            ),
            StartRequirement(
                RequirementId.BALANCED,
                "Gegenüberliegende Proben ausgewogen",
                observed.balanced,
                True,
                {**observed.source(), ManualField.FIELD: ManualField.BALANCED},
            ),
            StartRequirement(
                RequirementId.LID_CLOSED,
                "Deckel vollständig geschlossen",
                observed.lid_closed,
                True,
                {**observed.source(), ManualField.FIELD: ManualField.LID_CLOSED},
            ),
            StartRequirement(
                RequirementId.STOPPED,
                "Rotor steht still",
                observed.stopped,
                True,
                {**observed.source(), ManualField.FIELD: ManualField.RUNNING},
            ),
        ]
        allowed = all(requirement.satisfied is True for requirement in requirements)
        return {
            ManualField.ALLOWED: allowed,
            ManualField.REQUIREMENTS: [
                requirement.snapshot() for requirement in requirements
            ],
            ManualField.PLAN: self.plan(observed, allowed),
            ManualField.OBSERVATION_SOURCE: observed.source(),
        }

    def plan(self, observed: InstrumentObservation, allowed: bool) -> list[str]:
        """Explain the next operator steps for satisfying the assessed conditions."""
        if allowed:
            return ["Zentrifugenlauf starten."]
        if observed.available is not True:
            return ["Zentrifugenzustand aus der Simulation abfragen."]
        steps = []
        if observed.stopped is False:
            steps.append("Rotor stoppen und den Stillstand abwarten.")
        missing = (
            None
            if observed.secured_samples is None
            else self.active.minimum_samples - observed.secured_samples
        )
        needs_loading = missing is not None and missing > 0
        if missing is None:
            steps.append("Gesicherte Proben aus der Simulation abfragen.")
        elif needs_loading:
            steps.append(
                f"{missing} weitere Proben gegenüberliegend einsetzen und sichern."
            )
            if observed.lid_closed is True:
                steps.append(
                    "Dafür zuerst den Deckel öffnen; nach dem Sichern wieder schließen."
                )
        if observed.balanced is not True:
            steps.append("Masse gegenüberliegender Proben prüfen und ausgleichen.")
        if observed.lid_closed is not True:
            steps.append("Deckel vollständig schließen.")
        if observed.stopped is None:
            steps.append("Rotorstillstand aus der Simulation abfragen.")
        return steps

    def snapshot(self, instrument_snapshot: dict | None = None) -> dict[str, Any]:
        """Expose the active source, revision history and current derived conditions."""
        return {
            ManualField.MANUAL: self.active.snapshot(),
            ManualField.REVISION: self.revision,
            ManualField.CHANGES: copy.deepcopy(self.changes),
            ManualField.HISTORY: copy.deepcopy(list(self.history)),
            ManualField.EVALUATION: self.evaluate(instrument_snapshot or {}),
        }
