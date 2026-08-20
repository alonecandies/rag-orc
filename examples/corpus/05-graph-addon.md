# Graph Add-on

The Graph Add-on extends [Retrieval Engine](04-retrieval-engine.md) with entity and
relationship search. Where the base product ranks passages, the add-on builds a
graph of the entities those passages mention and lets a question be answered by
traversing it.

## Why a customer buys it

Two question shapes are effectively unanswerable by passage ranking alone.

The first is the **multi-hop** question, where the answer is not stated in any
single passage: *"who is on call for the service behind the product Contoso
bought?"* needs one document to connect the customer to a product, a second to
connect the product to a service, and a third to connect the service to a person.
Ranking passages by similarity to the question returns the three documents in some
order and leaves the join to the reader.

The second is the **aggregation** question — *"how many accounts run on the DACH
territory?"* — where the answer depends on the whole set rather than on the best
passage. Community summaries over the graph give a sensible answer where top-k
retrieval gives an arbitrary sample.

## How it is built

Ingest extracts entities and typed relationships per chunk, resolves the mentions
of one thing into one node, detects communities over the resolved graph, and writes
a summary per community. Entity resolution is the stage that decides whether any of
it works: without it a single customer arrives as three unconnected nodes and
traversal stops finding anything.

| Search mode | Entry point | Answers |
|---|---|---|
Local | entities named in the question | specific, entity-anchored questions |
Global | community summaries | corpus-wide and aggregation questions |
DRIFT | vector hits, then graph expansion | descriptive questions that name nothing |

## Pricing

It is billed at **$450 per month per seat**, and it cannot be bought on its own —
the environment must already be running version 3.0 or later. Seats are counted per
named user with graph query permission, not per environment, which is the one place
this product's metering differs from the platform's.

There is no usage component. Graph construction consumes
[Embedding Credits](06-embedding-credits.md) at the normal rate during ingest, and
the traversals themselves are free.

## Operations

The add-on runs in its own service, the **Graph Service**, backed by Neo4j. Tomas
Lindqvist owns it and is its on-call primary; the escalation ladder is in
[On-call](08-engineering-oncall.md). A Graph Service outage degrades queries to
passage ranking rather than failing them, so it is a SEV-2 under the definitions in
[Security](03-security-policy.md) and not a SEV-1.
