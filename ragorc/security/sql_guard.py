"""SQL validation for Text-to-SQL.

Threat model
------------
An LLM writes SQL from untrusted natural language, and we execute it. That makes
Text-to-SQL an arbitrary-query primitive; without a guard it is one prompt away
from ``DROP TABLE`` or ``COPY ... FROM PROGRAM``. Substring blocklists do not
close this — ``SELECT/*x*/ INTO``, ``sELeCt``, unicode homoglyphs and nested
CTEs all defeat them.

So validation happens on a **parsed AST** (sqlglot), where a statement type is a
node type and cannot be disguised by formatting. Three layers, in order:

1. **AST allowlist** — the statement must be a read, and no write/DDL/utility
   node may appear anywhere in the tree, at any depth.
2. **Semantic limits** — table allowlist, join ceiling, mandatory LIMIT,
   function blocklist.
3. **A read-only transaction and a SELECT-only database role** at execution time
   (see :mod:`ragorc.stores.postgres`). Layer 3 is what saves you when layers 1
   and 2 have a bug, which is why all three exist.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any

import sqlglot
import structlog
from sqlglot import exp

from ragorc.core.errors import GuardrailViolation
from ragorc.core.settings import SecuritySettings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["SQLGuard", "SQLValidation"]

#: Node types that mutate data, mutate schema, or reach outside the database.
#: Presence anywhere in the tree is fatal — including inside a CTE or subquery,
#: which is exactly where a naive top-level-only check gets bypassed.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Command,  # sqlglot's catch-all for COPY, VACUUM, CALL, SET, ...
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Use,
    exp.Merge,
    exp.Into,  # SELECT ... INTO new_table is a write
)

#: Functions that read the filesystem, open network connections, execute code or
#: burn wall-clock time. Blocked by name regardless of schema qualification.
_ALWAYS_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_logdir_ls",
        "lo_import",
        "lo_export",
        "lo_get",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "set_config",
        "current_setting",
        # The XML export family. Each one takes its target as a *string argument*,
        # so the AST's table and schema checks never see it: `table_to_xml(
        # 'pg_shadow', …)` reads a system catalog while referencing no table at
        # all, defeating the function blocklist, the table allowlist and the
        # catalog ban simultaneously. Blocked by name because argument inspection
        # cannot be made reliable — the argument may be a concatenation, a cast or
        # a parameter.
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "table_to_xmlschema",
        "table_to_xml_and_xmlschema",
        "schema_to_xml",
        "schema_to_xmlschema",
        "schema_to_xml_and_xmlschema",
        "database_to_xml",
        "database_to_xmlschema",
        "database_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        "xmlelement",
        "xmlforest",
        "xmlagg",
        "pg_read_server_files",
        "copy_from_program",
        "pg_file_write",
        "pg_file_unlink",
        "system",
        "shell",
        "exec",
    }
)

#: The ``*_to_xml*`` export family: ``query_to_xml``, ``table_to_xmlschema``,
#: ``database_to_xml_and_xmlschema`` and relatives. Matched by shape because the
#: family grows and every member is an exfiltration primitive that takes its
#: target as a string argument the AST cannot inspect.
_XML_EXPORT = re.compile(
    r"^(?:query|table|schema|database|cursor)_to_xml(?:schema|_and_xmlschema)?$"
)

#: System catalogs and metadata schemas. Reading them is how an attacker maps the
#: database before attacking it, and none of it belongs in a RAG answer.
_FORBIDDEN_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_toast"})


@dataclass(slots=True)
class SQLValidation:
    """Result of validating one statement."""

    sql: str
    """The rewritten, safe-to-execute statement (LIMIT injected if it was absent)."""
    tables: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    joins: int = 0
    has_limit: bool = False
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class SQLGuard:
    """Validates and normalizes LLM-generated SQL before execution."""

    def __init__(
        self,
        settings: SecuritySettings | None = None,
        *,
        dialect: str = "postgres",
        allowed_tables: list[str] | None = None,
        max_rows: int = 200,
    ) -> None:
        root = get_settings()
        self.settings = settings or root.security
        self.dialect = dialect
        self.allowed_tables = {t.lower() for t in (allowed_tables or root.postgres.allowed_tables)}
        self.max_rows = max_rows
        self.forbidden_functions = _ALWAYS_FORBIDDEN_FUNCTIONS | {
            f.lower() for f in self.settings.sql_forbid_functions
        }

    # -- public -----------------------------------------------------------
    def validate(self, sql: str, *, max_rows: int | None = None) -> SQLValidation:
        """Return a validated, LIMIT-bounded statement or raise.

        Raises :class:`GuardrailViolation`, which the retry decorator never
        retries — a blocked statement is still blocked the second time.
        """
        if not self.settings.enable_sql_guard:
            return SQLValidation(sql=sql, warnings=("guard disabled",))

        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            raise GuardrailViolation("empty SQL statement", rule="empty")

        self._reject_dangerous_characters(cleaned)

        try:
            parsed = sqlglot.parse(cleaned, dialect=self.dialect)
        except Exception as exc:  # sqlglot raises several parse error types
            raise GuardrailViolation(
                f"SQL failed to parse: {exc}", rule="parse_error", sql=cleaned[:300]
            ) from exc

        # `parse()` yields `None` for an empty statement (a stray `;`), so the
        # count below has to be taken after they are dropped. The annotation is
        # what carries the non-optional type into the rest of the guard: every
        # check below dereferences the node, and a `None` reaching them would be
        # an AttributeError inside a security check rather than a rejection.
        statements: list[exp.Expr] = [s for s in parsed if s is not None]
        if len(statements) != 1:
            # Statement stacking: "SELECT 1; DROP TABLE users" parses as two.
            raise GuardrailViolation(
                "exactly one statement is allowed",
                rule="multiple_statements",
                found=len(statements),
            )

        # The statement-type allowlist is also where the tree is narrowed: it
        # returns the node only if it is one of the concrete read nodes, all of
        # which are `exp.Expression` (the subclass of the `exp.Expr` trait that
        # carries `.args`). Everything downstream is checked against that.
        tree = self._check_statement_type(statements[0])
        self._check_projection(tree)
        self._check_forbidden_nodes(tree)
        self._check_functions(tree)
        tables = self._check_tables(tree)
        joins = self._check_joins(tree)
        columns = tuple(
            sorted({c.name for c in tree.find_all(exp.Column) if c.name and c.name != "*"})
        )

        warnings: list[str] = []
        limit = max_rows or self.max_rows
        bound = self._row_bound(tree)
        if bound is None:
            # No *provable* bound: either no LIMIT at all, or one whose value this
            # process cannot read (see _row_bound). Both are unbounded scans, and
            # both are handled the same way.
            if self.settings.sql_require_limit:
                tree = self._bound_rows(tree, limit)
                warnings.append(f"LIMIT {limit} injected")
        elif bound > limit:
            tree = self._bound_rows(tree, limit)
            warnings.append(f"LIMIT {bound} clamped to {limit}")

        if any(c.name == "*" for c in tree.find_all(exp.Star)):
            warnings.append("SELECT * returns unbounded columns; prefer explicit columns")

        safe_sql = tree.sql(dialect=self.dialect, comments=False)

        # Everything above validates the *input*. This validates the output, which
        # is a different statement: the guard rewrote it. A rewrite that produces
        # SQL the parser can no longer read is the guard's own bug, and letting it
        # through turns a clean refusal into a syntax error raised by the database
        # — past the point where the self-correction loop can do anything with it.
        # Found by the property tests, which re-parse what the guard returns.
        try:
            sqlglot.parse_one(safe_sql, dialect=self.dialect)
        except Exception as exc:
            raise GuardrailViolation(
                f"the guard rewrote this into SQL it can no longer parse: {exc}",
                rule="rewrite_unparseable",
                sql=safe_sql[:300],
            ) from exc

        # Re-derived from the *rewritten* tree, never asserted. `has_limit=True`
        # used to be hardcoded, so a statement the clamp had silently declined to
        # touch — `LIMIT NULL`, `LIMIT (SELECT 100000)`, `FETCH FIRST 999999 ROWS
        # ONLY` — came back claiming to be bounded. Callers use `.sql` directly,
        # and that flag is the only thing telling them whether it is safe to run.
        has_limit = self._row_bound(tree) is not None

        log.debug(
            "sql_validated", tables=tables, joins=joins, has_limit=has_limit, warnings=warnings
        )
        return SQLValidation(
            sql=safe_sql,
            tables=tables,
            columns=columns,
            joins=joins,
            has_limit=has_limit,
            warnings=tuple(warnings),
        )

    def is_safe(self, sql: str) -> bool:
        try:
            self.validate(sql)
        except GuardrailViolation:
            return False
        return True

    # -- checks -----------------------------------------------------------
    @staticmethod
    def _reject_dangerous_characters(sql: str) -> None:
        # A NUL byte truncates the statement inside libpq while our parser sees
        # the whole string — a classic parser/executor differential.
        if "\x00" in sql:
            raise GuardrailViolation("NUL byte in SQL", rule="nul_byte")
        # Bidi and invisible characters can hide a second statement from a human
        # reviewer while the server still executes it.
        for ch in ("‮", "‭", "⁦", "⁧", "⁨", "​", "‎", "‏"):
            if ch in sql:
                raise GuardrailViolation(
                    "bidirectional or zero-width character in SQL",
                    rule="unicode_control",
                    codepoint=hex(ord(ch)),
                )

    @staticmethod
    def _check_projection(tree: exp.Expression) -> None:
        """Reject a SELECT that selects nothing.

        ``SELECT`` alone parses — sqlglot builds a ``Select`` with an empty
        projection — so the statement-type allowlist waves it through, and the
        LIMIT clamp then renders ``SELECT LIMIT 100``, which no database will
        accept. A model that emits this has produced nothing worth running, and
        refusing it here gives the caller a ``GuardrailViolation`` it already
        handles instead of a driver-level syntax error.
        """
        for select in tree.find_all(exp.Select):
            if not select.expressions:
                raise GuardrailViolation("SELECT has an empty projection", rule="empty_projection")

    def _check_statement_type(self, tree: exp.Expr) -> exp.Expression:
        """Reject anything that is not a read, and return the accepted node.

        The return value is the input, unchanged; it exists so callers hold the
        node at its checked type. Accepting ``exp.Expr`` (sqlglot's trait base,
        which is what ``parse()`` is typed to produce) and returning
        ``exp.Expression`` makes the allowlist below the single place where an
        unverified parse result becomes a node the rest of the guard will walk.
        """
        allowed = {s.upper() for s in self.settings.sql_allow_statements}
        kind = type(tree).__name__.upper()
        # sqlglot models a top-level CTE as Select-with-With, and set operations
        # as Union/Intersect/Except; all are reads.
        read_nodes = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery, exp.With)
        if not isinstance(tree, read_nodes):
            raise GuardrailViolation(
                f"statement type {kind} is not permitted",
                rule="statement_type",
                allowed=sorted(allowed),
            )
        if "SELECT" not in allowed and isinstance(tree, exp.Select):
            raise GuardrailViolation(
                "SELECT is not in the allowed statement list", rule="statement_type"
            )
        if isinstance(tree, exp.With) and "WITH" not in allowed:
            raise GuardrailViolation(
                "WITH is not in the allowed statement list", rule="statement_type"
            )
        return tree

    @staticmethod
    def _check_forbidden_nodes(tree: exp.Expression) -> None:
        for node in tree.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise GuardrailViolation(
                    f"{type(node).__name__.upper()} is not permitted in a read-only query",
                    rule="forbidden_node",
                    node=type(node).__name__,
                )
            # Row locking (`FOR UPDATE`) takes write locks despite being a SELECT.
            if isinstance(node, exp.Lock):
                raise GuardrailViolation("row locking is not permitted", rule="lock")

    def _check_functions(self, tree: exp.Expression) -> None:
        for node in tree.find_all(exp.Func):
            name = (getattr(node, "sql_name", lambda: "")() or "").lower()
            if isinstance(node, exp.Anonymous):
                name = str(node.this).lower()
            if not name:
                continue
            lowered = name.lower()
            if lowered in self.forbidden_functions:
                raise GuardrailViolation(
                    f"function {name}() is not permitted", rule="forbidden_function", function=name
                )
            # Pattern check as well as the name list. The `*_to_xml*` family takes
            # its target as a string argument, so the AST checks cannot see what it
            # reads — and PostgreSQL keeps adding members. Matching the shape means
            # a new one is covered the day it ships rather than the day someone
            # notices, which for an exfiltration primitive is the difference that
            # matters.
            if _XML_EXPORT.match(lowered):
                raise GuardrailViolation(
                    f"function {name}() can export arbitrary tables and is not permitted",
                    rule="forbidden_function",
                    function=name,
                )

    def _check_tables(self, tree: exp.Expression) -> tuple[str, ...]:
        names: set[str] = set()
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        for table in tree.find_all(exp.Table):
            db = (table.text("db") or "").lower()
            if db in _FORBIDDEN_SCHEMAS:
                raise GuardrailViolation(
                    f"schema {db} is not readable", rule="forbidden_schema", schema=db
                )
            name = table.name.lower()
            if name.startswith("pg_"):
                raise GuardrailViolation(
                    f"system table {name} is not readable", rule="forbidden_table", table=name
                )
            # CTE references are not physical tables and must not be allowlisted.
            if name in cte_names:
                continue
            names.add(name)

        if self.allowed_tables:
            disallowed = names - self.allowed_tables
            if disallowed:
                raise GuardrailViolation(
                    "query references tables outside the allowlist",
                    rule="table_allowlist",
                    disallowed=sorted(disallowed),
                    allowed=sorted(self.allowed_tables),
                )
        return tuple(sorted(names))

    def _check_joins(self, tree: exp.Expression) -> int:
        joins = len(list(tree.find_all(exp.Join)))
        if joins > self.settings.sql_max_joins:
            raise GuardrailViolation(
                "too many joins", rule="join_limit", joins=joins, limit=self.settings.sql_max_joins
            )
        # A cross join over large tables is a denial-of-service by accident.
        for join in tree.find_all(exp.Join):
            side = (join.side or "").upper()
            kind = (join.kind or "").upper()
            if kind == "CROSS" or (
                not join.args.get("on") and not join.args.get("using") and not side
            ):
                log.warning("sql_cartesian_join", sql=tree.sql(dialect=self.dialect)[:200])
        return joins

    @staticmethod
    def _row_bound(tree: exp.Expression) -> int | None:
        """How many rows the statement can return, or ``None`` if unbounded.

        Only a plain integer literal counts as a bound, because only a plain
        integer literal *is* one:

        * ``LIMIT NULL`` is PostgreSQL's spelling of "no limit";
        * ``LIMIT (SELECT 100000)`` and ``LIMIT $1`` hold a value that is decided
          on the server, after validation;
        * ``FETCH FIRST n PERCENT`` / ``WITH TIES`` returns more rows than ``n``.

        The old test was ``args["limit"] is not None``, which answered "bounded"
        for every one of those — and the clamp, which read the value as an int,
        gave up and returned the tree untouched on exactly the same inputs. So the
        two disagreed silently: the guard reported a bound it had not applied.
        """
        target = tree.this if isinstance(tree, exp.Subquery) else tree
        node = target.args.get("limit")
        if node is None:
            return None
        if isinstance(node, exp.Fetch):
            # `FETCH FIRST n ROWS ONLY` occupies the same slot as LIMIT.
            options = node.args.get("limit_options")
            if options is not None and (
                options.args.get("percent") or options.args.get("with_ties")
            ):
                return None
            value = node.args.get("count")
        else:
            value = node.expression
        if isinstance(value, exp.Literal) and not value.is_string:
            try:
                return int(value.this)
            except (TypeError, ValueError):
                return None
        return None

    def _bound_rows(self, tree: exp.Expression, limit: int) -> exp.Expression:
        """Force the outermost result to at most ``limit`` rows.

        ``limit()`` overwrites whatever occupied the slot, which is the point:
        ``LIMIT NULL`` and ``FETCH FIRST 999999 ROWS ONLY`` are replaced by a
        plain integer rather than left in place beside a second clause.
        """
        # Anything this block cannot rewrite in place — an odd shape that raises,
        # or a node with no LIMIT slot at all — falls through to the wrapper.
        with contextlib.suppress(Exception):
            if isinstance(tree, exp.Subquery) and tree.this is not None:
                # `(SELECT … LIMIT 999999)`: the limit that governs the scan is
                # inside the wrapper. Calling limit() on the Subquery bolts a
                # second one on the outside and leaves the inner one as written,
                # which reads as bounded while the plan is not.
                out = tree.copy()
                out.set("this", out.this.limit(limit))
                return out
            if isinstance(tree, exp.Query):
                # `limit()` is defined on the Query trait (Select/Union/Intersect/
                # Except/Subquery). A node without it — `With` is the one the
                # allowlist admits — has no LIMIT slot to overwrite, and reached
                # the wrapper via an AttributeError before this check drew the
                # same line up front.
                return tree.limit(limit)
        wrapped = sqlglot.parse_one(  # pragma: no cover - odd shapes need a wrapper
            # Not injection: the inner SQL is re-serialized from a validated
            # AST and `limit` is an int from settings.
            f"SELECT * FROM ({tree.sql(dialect=self.dialect)}) AS _guarded LIMIT {limit}",  # noqa: S608
            dialect=self.dialect,
        )
        # `parse_one` is typed to the `Expr` trait; a SELECT always parses to a
        # concrete `Select`. Asserted rather than cast: this is the one path that
        # hands back a tree nobody re-checks, so a surprise here should stop the
        # query, not travel on as an unbounded statement.
        assert isinstance(wrapped, exp.Expression)
        return wrapped
