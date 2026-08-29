"""What the generator is actually handed when a query is constructed for it.

Round eight audited the SQL and Cypher *guards*. Nobody had audited what happens
to a result between the store executing it and the model reading it, and three
renderers turned out to be unreachable for one shared reason: **both stores
normalize result values for JSON-safety, and that normalization runs before the
construct module's renderers see them.** Each store's conversion is right for its
own purpose and wrong as a preprocessing step for code written to handle the raw
types.

The worst of the three did not merely render badly — it returned the wrong rows.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ragorc.construct.text_to_cypher import _render_value
from ragorc.construct.text_to_sql import _cell


# ---------------------------------------------------------------------------
# Duplicate column names
# ---------------------------------------------------------------------------
def test_duplicate_column_names_are_disambiguated_not_dropped() -> None:
    """`SELECT * FROM a JOIN b` is the commonest shape a text-to-SQL model
    produces, and both tables usually carry `id`. `dict_row` kept the *last* of
    each, so a six-column result arrived as three columns holding the right-hand
    table's values under the left-hand table's question — no error, and nothing
    in the row count to notice.
    """
    from ragorc.stores.postgres.store import _disambiguate

    assert _disambiguate(["id", "customer", "amount", "id", "customer", "amount"]) == [
        "id",
        "customer",
        "amount",
        "id__2",
        "customer__2",
        "amount__2",
    ]


def test_the_first_occurrence_keeps_the_bare_name() -> None:
    """A query reads left to right, so renaming the column the author meant would
    be the more surprising half of the fix."""
    from ragorc.stores.postgres.store import _disambiguate

    assert _disambiguate(["total", "total", "total"]) == ["total", "total__2", "total__3"]


def test_distinct_names_are_untouched() -> None:
    from ragorc.stores.postgres.store import _disambiguate

    names = ["a", "b", "c"]
    assert _disambiguate(names) == names


# ---------------------------------------------------------------------------
# Decimal scale
# ---------------------------------------------------------------------------
def test_a_decimal_keeps_its_scale_through_the_store() -> None:
    """`_json_safe` called `float()` on the grounds that it was "the honest lossy
    rendering". It is not honest — it changes the digits — and the value lands in
    the one chunk class the generator is told to reproduce verbatim."""
    from ragorc.stores.postgres.store import _json_safe

    assert _json_safe(Decimal("12.50")) == "12.50"
    assert _json_safe(Decimal("1234567890123456789.99")) == "1234567890123456789.99"


def test_the_cell_renderer_agrees_with_the_store() -> None:
    """The two used to contradict each other in comments, and the store won by
    position: `_cell`'s Decimal branch was the only one that preserved scale and
    could never run."""
    assert _cell(Decimal("12.50")) == "12.50"

    from ragorc.stores.postgres.store import _json_safe

    assert _cell(_json_safe(Decimal("12.50"))) == "12.50"


def test_floats_and_ints_are_still_numbers() -> None:
    """Only Decimal changes: a float column must not start arriving as a string."""
    from ragorc.stores.postgres.store import _json_safe

    assert _json_safe(1.5) == 1.5
    assert _json_safe(3) == 3
    assert _json_safe(True) is True


# ---------------------------------------------------------------------------
# The graph verbalizer
# ---------------------------------------------------------------------------
def _serialized_path() -> dict[str, object]:
    """Exactly what `Neo4jStore._serialize` emits for a two-hop path."""
    return {
        "_nodes": [
            {"_element_id": "4:x:183", "_labels": ["Company"], "name": "Northwind"},
            {"_element_id": "4:x:184", "_labels": ["Company"], "name": "Contoso"},
            {"_element_id": "4:x:185", "_labels": ["Company"], "name": "Fabrikam"},
        ],
        "_relationships": [
            {"_element_id": "5:x:1", "_type": "SUPPLIES", "_start": "4:x:183", "_end": "4:x:184"},
            {"_element_id": "5:x:2", "_type": "OWNED_BY", "_start": "4:x:184", "_end": "4:x:185"},
        ],
        "_length": 2,
    }


def test_a_serialized_path_is_verbalized() -> None:
    """The module docstring's whole justification: "handed to a model as a repr,
    they are unreadable noise". The store flattens `Path` to a dict before this
    code sees it, and every detector here probes for *attributes* — `.nodes`,
    `.labels`, `.start_node` — which a dict does not have however many matching
    keys it carries. So the generator received the element ids.
    """
    assert _render_value(_serialized_path()) == (
        "Northwind -[SUPPLIES]-> Contoso -[OWNED_BY]-> Fabrikam"
    )


def test_a_serialized_node_renders_as_label_and_name() -> None:
    rendered = _render_value(
        {"_element_id": "4:x:183", "_labels": ["Company"], "name": "Northwind", "sector": "logistics"}
    )
    assert rendered == "Company(Northwind)[sector=logistics]"
    assert "_element_id" not in rendered, "internal ids are meaningless outside the database"


def test_a_bare_relationship_renders_without_inventing_endpoints() -> None:
    """`_start`/`_end` are element ids, not nodes. Outside a path there is nothing
    to resolve them against, so the honest rendering names neither — which is what
    `_render_relationship` already does for missing endpoints."""
    assert _render_value(
        {"_element_id": "5:x:1", "_type": "SUPPLIES", "_start": "4:x:183", "_end": "4:x:184"}
    ) == "-[SUPPLIES]->"


def test_a_plain_dict_is_still_a_plain_dict() -> None:
    """The adapter must only claim rows it recognizes: a scalar projection —
    `RETURN n.name AS name, count(*) AS n` — is a dict too."""
    assert _render_value({"a": 1, "b": "two"}) == "{a=1, b=two}"


def test_real_driver_objects_still_render() -> None:
    """The adapter is additive. A caller handing `to_chunks` raw records — which
    is what the renderers were written for — must keep working, or the fix trades
    one dead path for another.
    """

    class Node:
        def __init__(self, name: str) -> None:
            self.labels = ("Company",)
            self._props = {"name": name}

        def items(self) -> object:
            return self._props.items()

    class Rel:
        type = "SUPPLIES"

        def __init__(self, start: object, end: object) -> None:
            self.start_node = start
            self.end_node = end

    a, b = Node("Northwind"), Node("Contoso")

    class Path:
        nodes = (a, b)
        relationships = (Rel(a, b),)

    assert _render_value(Path()) == "Northwind -[SUPPLIES]-> Contoso"


@pytest.mark.parametrize("missing", ["_nodes", "_relationships"])
def test_a_half_formed_path_does_not_raise(missing: str) -> None:
    """Defensive: the adapter reads whatever the driver's serializer produced, and
    a query returning an empty path must not take the answer down."""
    data = _serialized_path()
    data.pop(missing)
    _render_value(data)  # must not raise


def test_a_verbalized_path_is_not_prefixed_with_its_return_alias() -> None:
    """The same attribute-versus-key mismatch, one frame up.

    `_render_row` decides whether to prefix a value with its RETURN alias by
    asking `_is_path_like` — and it asked the *raw* value, so after the renderers
    were adapted a path came out correct and still wearing `p: `. The comment
    beside that line says the prefix "only adds noise" on a path, which is
    precisely what it kept adding.
    """
    from ragorc.construct.text_to_cypher import _render_row

    assert _render_row({"p": _serialized_path()}) == (
        "Northwind -[SUPPLIES]-> Contoso -[OWNED_BY]-> Fabrikam"
    )


def test_a_scalar_projection_keeps_its_aliases() -> None:
    """`RETURN n.name AS name, count(*) AS n` is not a sentence, and the aliases
    are what make the row readable."""
    from ragorc.construct.text_to_cypher import _render_row

    assert _render_row({"name": "Northwind", "n": 3}) == "name: Northwind | n: 3"
