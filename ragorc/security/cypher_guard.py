"""Cypher validation for Text-to-Cypher.

Unlike SQL there is no maintained Python Cypher parser, so validation is
lexical — but done properly, which means *normalizing before scanning*.

The naive version (``if "DELETE" in cypher.upper()``) fails two ways at once:
it rejects ``MATCH (n) WHERE n.name = 'DELETE ME' RETURN n`` (a false positive
that breaks legitimate queries) and it accepts ``MATCH (n) /*x*/ DETACH  DELETE
n`` in some formulations. So the guard first strips comments and string literals,
then scans the remainder for whole-word keywords.

Three Cypher-specific hazards get their own checks:

* **Unbounded variable-length patterns.** ``MATCH (a)-[*]-(b)`` walks the entire
  graph. On any real dataset that is an outage, not a slow query. The scan is
  anchored on the relationship bracket, because a ``*`` on its own is far more
  often ``count(*)`` than a hop range.
* **Procedure calls.** ``CALL apoc.load.json('http://…')`` is server-side request
  forgery and ``CALL dbms.*`` exposes administration. Procedures are therefore
  allowlisted, not blocklisted. Being on the allowlist settles whether a
  procedure may run, not how far it walks, so the depth argument of the
  ``apoc.path.*`` family is checked against the same hop ceiling.
* **Cypher passed as a function argument.** ``apoc.cypher.runFirstColumn(q, {})``
  executes ``q``. Every check above reads the *outer* query, so a nested query
  smuggled through an argument is invisible to all of them; that family is
  refused by name wherever it appears, ``CALL`` or not.

An optional ``EXPLAIN`` dry run adds a final check the server itself performs:
it catches syntax errors and, because Neo4j reports the plan, wildly expensive
plans — without touching data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from ragorc.core.errors import GuardrailViolation
from ragorc.core.settings import SecuritySettings, get_settings

if TYPE_CHECKING:
    # Guarded because ragorc.core.protocols pulls in ragorc.core.models, and the
    # guard is imported by the stores that satisfy the protocol. Structural
    # typing needs the name only at check time.
    from ragorc.core.protocols import GraphStore

log = structlog.get_logger(__name__)

__all__ = ["CypherGuard", "CypherValidation"]

#: Clauses that mutate. ``FOREACH`` is included because its body can contain
#: writes, and ``USE`` because it can redirect the query to another database.
#:
#: This list is the floor, not a default: ``__init__`` unions it with
#: ``settings.security.cypher_forbid_keywords`` rather than letting the setting
#: replace it. The shipped setting is a non-empty list that omits ``USE``,
#: ``FOREACH``, ``START``/``STOP DATABASE``, ``ALTER``, ``RENAME`` and ``ENABLE``,
#: so the old ``setting or _WRITE_KEYWORDS`` meant this tuple never ran and the
#: docstring above described protection nobody had — ``USE system MATCH (n)
#: RETURN n`` validated clean. Union, not override: an operator narrowing the
#: setting is tuning what *else* to refuse, not opting out of read-only.
_WRITE_KEYWORDS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "FOREACH",
    "LOAD CSV",
    "USE",
    "GRANT",
    "DENY",
    "REVOKE",
    "START DATABASE",
    "STOP DATABASE",
    "TERMINATE",
    "ALTER",
    "RENAME",
    "ENABLE",
)

#: Read-only procedures we permit. Anything else is refused: the set of harmful
#: procedures is open-ended, the set of ones we need is small.
_ALLOWED_PROCEDURES = frozenset(
    {
        "db.labels",
        "db.relationshiptypes",
        "db.propertykeys",
        "db.schema.visualization",
        "db.schema.nodetypeproperties",
        "db.schema.reltypeproperties",
        "db.index.fulltext.querynodes",
        "db.index.fulltext.queryrelationships",
        "db.index.vector.querynodes",
        "db.indexes",
        "db.constraints",
        "apoc.meta.stats",
        "apoc.meta.schema",
        "apoc.path.expand",
        "apoc.path.expandconfig",
        "apoc.path.subgraphall",
        "apoc.path.subgraphnodes",
        "apoc.algo.dijkstra",
        "apoc.coll.subtract",
        "apoc.text.levenshteinsimilarity",
    }
)

_STRING_OR_COMMENT = re.compile(
    r"""
    '(?:[^'\\]|\\.)*'          # single-quoted literal
  | "(?:[^"\\]|\\.)*"          # double-quoted literal
  | `(?:[^`]|``)*`             # backtick-quoted identifier
  | //[^\n]*                   # line comment
  | /\*.*?\*/                  # block comment
    """,
    re.VERBOSE | re.DOTALL,
)

#: A hop range only exists inside a *relationship* bracket, and in Cypher a
#: relationship bracket is always attached to a dash: ``-[r*1..3]->``, ``<-[*]-``.
#: The hop scan therefore runs on bracket contents only.
#:
#: It used to run on the whole query, which meant every ``*`` was read as a hop
#: range: ``RETURN count(*)`` was rejected as an unbounded traversal (the ``*)``
#: matched the bare-star branch) and ``RETURN n.price * 100`` as a 100-hop
#: pattern. That is worse than it sounds — it made the graph leg unable to answer
#: the counting questions it exists for, and a guard that refuses ordinary work is
#: a guard an operator switches off, taking the real checks with it.
_REL_BRACKET = re.compile(r"-\s*\[([^\]]*)\]")
#: Applied to the contents of a relationship bracket: matches *, *.., *2..,
#: :TYPE*, r*1.. — any hop range without an upper bound.
_UNBOUNDED_HOPS = re.compile(r"\*\s*(?:\d+\s*\.\.\s*(?!\d)|\.\.\s*(?!\d)|(?!\s*\d))")
_HOP_RANGE = re.compile(r"\*\s*(\d+)?\s*(?:\.\.\s*(\d+))?")

#: A quantified path pattern repeats a whole path: ``((a)-[r]->(b))*``, ``(…)+``,
#: ``(…){2,}`` are all unbounded traversals spelled *outside* any bracket, so
#: anchoring the scan on brackets would have dropped the one non-bracket shape
#: that really is a hop range. ``count(*)`` is ``(``-``*``-``)`` and can never
#: match this, and arithmetic such as ``(a.x - a.y) * 2`` is excluded by requiring
#: the quantified group to contain a relationship of its own.
_PATH_QUANTIFIER = re.compile(r"\)\s*(?:\*|\+|\{\s*\d*\s*,\s*\})")
_RELATIONSHIP = re.compile(r"-\s*\[|->|<-")
#: How far back ``_quantified_path`` looks for the ``(`` a quantifier belongs to.
_QUANTIFIER_LOOKBACK = 2000

#: ``apoc.cypher.run``, ``.runFirstColumn``, ``.runFirstColumnSingle``,
#: ``.runFirstColumnMany``, ``.doIt``, ``.runWrite`` … every member of this
#: namespace takes a Cypher string as an *argument* and executes it. Nothing in
#: it is safe on a read-only path, so the whole namespace goes by prefix rather
#: than by member: the list grows with APOC releases, and the failure mode of
#: missing a member is arbitrary Cypher execution.
#:
#: The allowlist already refused ``CALL apoc.cypher.runFirstColumn(…)``. The hole
#: was the *function* spelling — ``RETURN apoc.cypher.runFirstColumnSingle('MATCH
#: (a)-[*]-(b) RETURN count(*)', {})`` has no CALL target, no write keyword and no
#: bracket outside a string literal, so every check read the outer query and found
#: nothing while the inner unbounded traversal ran.
_NESTED_CYPHER = re.compile(r"\bapoc\s*\.\s*cypher\s*\.\s*\w+", re.IGNORECASE)

#: The path-expansion procedures on the allowlist take their depth as an
#: *argument*, and APOC spells "unlimited" as ``-1`` — which is also what
#: ``maxLevel`` defaults to when the config map omits it. So ``CALL
#: apoc.path.expand(n, null, null, 0, -1)`` is the same whole-graph walk as
#: ``-[*]-`` wearing an allowlisted name, and the bracket scan cannot see it
#: because there is no bracket. The depth argument is therefore read and held to
#: the same ceiling, and a call whose depth this process cannot read counts as
#: unbounded — the call ``_row_bound`` makes in the SQL guard, for the same
#: reason: an unprovable bound is not a bound.
_APOC_PATH_CALL = re.compile(r"\bapoc\s*\.\s*path\s*\.\s*(\w+)\s*\(", re.IGNORECASE)
_MAX_LEVEL = re.compile(r"\bmaxLevel\s*:\s*(-?\d+)", re.IGNORECASE)
#: Only ``apoc.path.expand`` takes its depth positionally, as the last of five.
#: Its relatives take a config map, where the depth is keyed.
_POSITIONAL_DEPTH = {"expand": 5}
_INTEGER = re.compile(r"-?\d+")

#: ``CALL`` targets, allowing the backtick-quoted spelling Neo4j accepts.
#:
#: The unquoted-only version was a live bypass:
#: ``CALL `apoc.load.json`('http://169.254.169.254/')`` matched nothing, so the
#: allowlist never ran and the query reached the database — server-side request
#: forgery against the cloud metadata endpoint, from inside the database. It was
#: doubly hidden because ``_STRING_OR_COMMENT`` blanks backtick-quoted
#: identifiers before the scan, which is why the procedure scan now runs against
#: a separately normalized copy (see ``_normalize_call_targets``).
_CALL_PROCEDURE = re.compile(r"\bCALL\s+`?([A-Za-z_][\w.]*)`?", re.IGNORECASE)

#: Administrative ``SHOW`` commands. Read-only, and still not for a RAG query
#: path: they enumerate users, roles, configuration and live transactions.
_SHOW_ADMIN = re.compile(
    r"\bSHOW\s+(?:CURRENT\s+)?"
    r"(USERS?|ROLES?|PRIVILEGES?|SETTINGS?|DATABASES?|DATABASE|SERVERS?|"
    r"TRANSACTIONS?|PROCEDURES?|FUNCTIONS?|CONSTRAINTS?|INDEXES?|ALIASES?)\b",
    re.IGNORECASE,
)


_COMMENT_OR_STRING_ONLY = re.compile(
    r"""
    '(?:[^'\\]|\\.)*'
  | "(?:[^"\\]|\\.)*"
  | //[^\n]*
  | /\*.*?\*/
    """,
    re.VERBOSE | re.DOTALL,
)


def _normalize_call_targets(cypher: str) -> str:
    r"""Blank comments and strings, but keep backtick-quoted identifiers visible.

    The main scan blanks backticks along with strings, which is correct for
    keyword matching — a label called ``\`DELETE\``` is not a clause. It is wrong
    for the procedure allowlist, because Neo4j accepts a backtick-quoted procedure
    name and the allowlist then matched nothing at all.
    """
    stripped = _COMMENT_OR_STRING_ONLY.sub(lambda m: " " * len(m.group(0)), cypher)
    # Unwrap the quoting so the allowlist compares like with like.
    return stripped.replace("`", "")


@dataclass(slots=True)
class CypherValidation:
    cypher: str
    has_limit: bool = False
    max_hops: int | None = None
    procedures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class CypherGuard:
    """Validates and normalizes LLM-generated Cypher before execution."""

    def __init__(
        self,
        settings: SecuritySettings | None = None,
        *,
        max_rows: int = 200,
        max_hops: int = 4,
        allowed_labels: set[str] | None = None,
    ) -> None:
        self.settings = settings or get_settings().security
        self.max_rows = max_rows
        self.max_hops = max_hops
        self.allowed_labels = allowed_labels
        # Union, not `setting or _WRITE_KEYWORDS`: see the note on
        # _WRITE_KEYWORDS. Order-preserving dedup so the reported keyword stays
        # stable between runs.
        self.forbidden = tuple(
            dict.fromkeys(
                k.upper() for k in (*_WRITE_KEYWORDS, *self.settings.cypher_forbid_keywords)
            )
        )

    # -- public -----------------------------------------------------------
    def validate(self, cypher: str, *, max_rows: int | None = None) -> CypherValidation:
        if not self.settings.enable_cypher_guard:
            return CypherValidation(cypher=cypher, warnings=("guard disabled",))

        raw = cypher.strip().rstrip(";").strip()
        if not raw:
            raise GuardrailViolation("empty Cypher statement", rule="empty")
        if "\x00" in raw:
            raise GuardrailViolation("NUL byte in Cypher", rule="nul_byte")

        # Scanning target: literals and comments blanked out so their contents
        # can neither trigger a false positive nor hide a real keyword.
        scan = _STRING_OR_COMMENT.sub(lambda m: " " * len(m.group(0)), raw)

        if ";" in scan.strip().rstrip(";"):
            raise GuardrailViolation(
                "multiple statements are not permitted", rule="multiple_statements"
            )

        self._check_keywords(scan)
        # Procedures are scanned on a copy that preserves backtick-quoted
        # identifiers: the general blanking pass removes them (they are
        # identifiers, not clauses), which is exactly what hid the backtick
        # bypass. Comments and *string* literals are still blanked here.
        call_scan = _normalize_call_targets(raw)
        procedures = self._check_procedures(call_scan)
        # After the allowlist, so a `CALL apoc.cypher.*` keeps reporting
        # `forbidden_procedure` and only the function spelling — which nothing
        # caught — is attributed to this check.
        self._check_nested_cypher(call_scan)
        self._check_show(scan)
        # Both spellings of a hop ceiling, in one place: brackets read from the
        # main scan, procedure depth arguments from the copy that still shows
        # backtick-quoted names — `CALL `apoc.path.expand`(…, -1)` is allowlisted
        # once the backticks come off, so the depth check has to see it too.
        max_hops = self._check_hops(scan, call_scan)
        self._require_read_clause(scan)

        warnings: list[str] = []
        limit = max_rows or self.max_rows
        has_limit = re.search(r"\bLIMIT\s+\d+", scan, re.IGNORECASE) is not None
        out = raw
        if has_limit:
            out, clamped = self._clamp_limit(raw, scan, limit)
            if clamped:
                warnings.append(f"LIMIT clamped to {limit}")
        else:
            out = f"{raw}\nLIMIT {limit}"
            warnings.append(f"LIMIT {limit} appended")

        log.debug("cypher_validated", procedures=procedures, max_hops=max_hops, warnings=warnings)
        return CypherValidation(
            cypher=out,
            has_limit=True,
            max_hops=max_hops,
            procedures=procedures,
            warnings=tuple(warnings),
        )

    def is_safe(self, cypher: str) -> bool:
        try:
            self.validate(cypher)
        except GuardrailViolation:
            return False
        return True

    async def explain(self, store: GraphStore, cypher: str) -> dict[str, Any]:
        """Server-side dry run. EXPLAIN compiles and plans without executing, so
        it validates syntax and surfaces the plan for free."""
        if not self.settings.cypher_explain_dryrun:
            return {}
        try:
            rows = await store.execute_readonly(f"EXPLAIN {cypher}", limit=1)
        except Exception as exc:  # noqa: BLE001 - surface as a guard failure
            raise GuardrailViolation(
                f"Cypher failed to plan: {exc}", rule="explain_failed", cypher=cypher[:300]
            ) from exc
        return {"explained": True, "rows": rows}

    # -- checks -----------------------------------------------------------
    def _check_keywords(self, scan: str) -> None:
        upper = scan.upper()
        for keyword in self.forbidden:
            # Multi-word keywords ("LOAD CSV", "DETACH DELETE") may be separated
            # by arbitrary whitespace or newlines in generated Cypher.
            #
            # `(?<!\.)` keeps a property read out of the scan. A clause keyword is
            # never preceded by a dot in Cypher, but a property is: unioning
            # _WRITE_KEYWORDS in added USE, ENABLE, ALTER and RENAME, which are
            # ordinary property names, and `RETURN n.use` must not be read as a
            # `USE` clause. Nothing is weakened — `n.delete` is a property read,
            # and there is no syntax that turns it into a write.
            pattern = (
                r"(?<!\.)\b" + r"\s+".join(re.escape(part) for part in keyword.split()) + r"\b"
            )
            if re.search(pattern, upper):
                raise GuardrailViolation(
                    f"{keyword} is not permitted in a read-only query",
                    rule="forbidden_keyword",
                    keyword=keyword,
                )

    @staticmethod
    def _check_procedures(scan: str) -> tuple[str, ...]:
        found: list[str] = []
        for match in _CALL_PROCEDURE.finditer(scan):
            name = match.group(1)
            lowered = name.lower()
            if lowered not in _ALLOWED_PROCEDURES:
                raise GuardrailViolation(
                    f"procedure {name} is not on the read-only allowlist",
                    rule="forbidden_procedure",
                    procedure=name,
                    allowed=sorted(_ALLOWED_PROCEDURES)[:10],
                )
            found.append(lowered)
        return tuple(found)

    @staticmethod
    def _check_nested_cypher(scan: str) -> None:
        """Refuse the ``apoc.cypher.*`` family wherever it appears.

        Not only after ``CALL``: as a function in a ``RETURN``/``WITH`` it reads
        as an ordinary projection, and the Cypher it executes lives in a string
        literal that every other check has already blanked out. So the guard's
        keyword scan, procedure allowlist and hop ceiling all pass while the
        nested query does whatever it likes — the hop limit was demonstrably
        bypassable this way.
        """
        match = _NESTED_CYPHER.search(scan)
        if match:
            name = match.group(0)
            raise GuardrailViolation(
                f"function {name}() executes a nested Cypher string and is not permitted",
                rule="forbidden_function",
                function=name,
            )

    def _check_path_depth(self, scan: str) -> int | None:
        """Hold ``apoc.path.*`` to the hop ceiling its arguments carry.

        The allowlist answers "may this procedure run", never "how far does it
        walk", so ``apoc.path.expand`` — allowlisted, and rightly, since bounded
        expansion is what the graph leg is for — was an unbounded traversal with
        a ``-1`` in the fifth argument. Nothing else caught it: no bracket, no
        write keyword, no nested Cypher string.

        Matched wherever the name appears rather than only after ``CALL``, for
        the reason ``_check_nested_cypher`` gives: the function spelling reads as
        an ordinary projection and every check that scans the outer query misses
        it.
        """
        widest: int | None = None
        for match in _APOC_PATH_CALL.finditer(scan):
            name = match.group(1).lower()
            arguments = self._call_arguments(scan, match.end() - 1)
            depth = None if arguments is None else self._path_depth(name, arguments)
            if depth is None or depth < 0:
                raise GuardrailViolation(
                    f"apoc.path.{name} must carry a readable maxLevel no greater than "
                    f"{self.max_hops}; APOC's default is -1, which walks the whole graph",
                    rule="unbounded_hops",
                    procedure=f"apoc.path.{name}",
                )
            if depth > self.max_hops:
                raise GuardrailViolation(
                    "path expansion exceeds the hop limit",
                    rule="hop_limit",
                    hops=depth,
                    limit=self.max_hops,
                )
            # Every call, not the first: one bounded expansion beside an
            # unbounded one is still an unbounded query.
            widest = depth if widest is None else max(widest, depth)
        return widest

    @staticmethod
    def _call_arguments(scan: str, open_paren: int) -> str | None:
        """The text inside the parentheses opening at ``open_paren``.

        ``None`` when they never close, which the caller reads as "no provable
        bound" rather than as "nothing to check" — so an argument list this
        process cannot delimit costs the query, not the ceiling.
        """
        depth = 0
        for index in range(open_paren, len(scan)):
            char = scan[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return scan[open_paren + 1 : index]
        return None

    @staticmethod
    def _path_depth(name: str, arguments: str) -> int | None:
        """The depth ``apoc.path.<name>(arguments)`` will walk, if it is readable.

        ``maxLevel`` wins wherever it appears, because every config-map form
        keys it that way. Only the positional signature is read by position, and
        only when the argument count matches — guessing at a short-form call
        would either invent a bound that is not there or refuse a legal one.
        A ``limit:`` or ``minLevel:`` in the same map is not a depth and is left
        alone; reading every integer in the argument list would turn
        ``{maxLevel: 2, limit: 500}`` into a hop-limit rejection.
        """
        keyed = _MAX_LEVEL.search(arguments)
        if keyed:
            return int(keyed.group(1))
        expected = _POSITIONAL_DEPTH.get(name)
        if expected is None:
            return None
        parts = CypherGuard._split_arguments(arguments)
        if len(parts) != expected:
            return None
        last = parts[-1].strip()
        return int(last) if _INTEGER.fullmatch(last) else None

    @staticmethod
    def _split_arguments(arguments: str) -> list[str]:
        """Split on commas that are not inside a nested bracket of any kind.

        String literals are already blanked in the text this runs on, so only
        nesting can hide a comma.
        """
        parts: list[str] = []
        depth = 0
        start = 0
        for index, char in enumerate(arguments):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(arguments[start:index])
                start = index + 1
        parts.append(arguments[start:])
        return parts

    @staticmethod
    def _quantified_path(scan: str) -> bool:
        """True when an unbounded quantifier applies to a whole path pattern.

        The backward walk is what separates ``((a)-[r]->(b))*`` from ``(a.x -
        a.y) * 2``: it finds the ``(`` the quantifier belongs to and asks whether
        that group contains a relationship. A fixed look-behind window instead
        would call ``MATCH (a)-[r]->(b) RETURN (b.price - a.price) * 2`` a
        traversal, which is the false-positive class this check was rewritten to
        escape.

        The walk is bounded because it runs on attacker-supplied text: unbalanced
        parentheses make each candidate scan back to the start, which is quadratic
        (6 KB of ``)*`` took 280 ms). Past the bound the group is unprovable, and
        unprovable means unbounded — the same call ``_row_bound`` makes in the SQL
        guard. No legitimate quantified group is 2 KB wide.
        """
        for match in _PATH_QUANTIFIER.finditer(scan):
            depth = 0
            floor = max(0, match.start() - _QUANTIFIER_LOOKBACK)
            for index in range(match.start(), floor - 1, -1):
                char = scan[index]
                if char == ")":
                    depth += 1
                elif char == "(":
                    depth -= 1
                    if depth == 0:
                        if _RELATIONSHIP.search(scan[index : match.start()]):
                            return True
                        break  # arithmetic, not a path
            else:
                return True  # never resolved: treat as a path, fail closed
        return False

    def _check_hops(self, scan: str, call_scan: str | None = None) -> int | None:
        # Only relationship brackets and quantified path patterns are hop
        # ranges. A `CALL { ... }` subquery block is fine, and so are `count(*)`,
        # `collect(*)` and `n.price * 100`, all of which this check used to
        # reject.
        widest: int | None = self._check_path_depth(call_scan if call_scan is not None else scan)
        brackets = [m.group(1) for m in _REL_BRACKET.finditer(scan)]
        if self._quantified_path(scan) or any(_UNBOUNDED_HOPS.search(b) for b in brackets):
            raise GuardrailViolation(
                "variable-length relationship patterns must have an upper bound "
                "(use *1..3, never *)",
                rule="unbounded_hops",
            )
        for match in (m for bracket in brackets for m in _HOP_RANGE.finditer(bracket)):
            upper = match.group(2) or match.group(1)
            if upper is None:
                continue
            value = int(upper)
            widest = value if widest is None else max(widest, value)
        if widest is not None and widest > self.max_hops:
            raise GuardrailViolation(
                "relationship pattern exceeds the hop limit",
                rule="hop_limit",
                hops=widest,
                limit=self.max_hops,
            )
        return widest

    @staticmethod
    def _check_show(scan: str) -> None:
        """Refuse administrative ``SHOW`` commands.

        They are read-only, so the write-keyword scan lets them through, and
        ``SHOW SETTINGS YIELD name RETURN name`` even satisfies the RETURN
        requirement. None of them answer a question about the corpus; all of them
        describe the deployment. ``SHOW USERS`` enumerates accounts and
        ``SHOW SETTINGS`` leaks configuration, so a question-answering path has no
        business reaching either.
        """
        match = _SHOW_ADMIN.search(scan)
        if match:
            raise GuardrailViolation(
                f"SHOW {match.group(1).upper()} is administrative and not permitted",
                rule="forbidden_show",
                command=match.group(0),
            )

    @staticmethod
    def _require_read_clause(scan: str) -> None:
        upper = scan.upper()
        if not re.search(r"\b(MATCH|RETURN|UNWIND|WITH|CALL|SHOW|EXPLAIN|PROFILE)\b", upper):
            raise GuardrailViolation("query contains no read clause", rule="no_read_clause")
        if not re.search(r"\b(RETURN|YIELD)\b", upper):
            raise GuardrailViolation("query must RETURN (or YIELD) something", rule="no_return")

    @staticmethod
    def _clamp_limit(raw: str, scan: str, limit: int) -> tuple[str, bool]:
        """Rewrite an over-large LIMIT, operating on offsets found in the
        blanked-out scan text so a literal like '"LIMIT 100"' is never touched."""
        clamped = False
        out = raw
        for match in reversed(list(re.finditer(r"\bLIMIT\s+(\d+)", scan, re.IGNORECASE))):
            value = int(match.group(1))
            if value > limit:
                start, end = match.span(1)
                out = out[:start] + str(limit) + out[end:]
                clamped = True
        return out, clamped
