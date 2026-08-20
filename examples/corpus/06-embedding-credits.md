# Embedding Credits

The usage-metered product. Every document that enters an index and every query that
leaves it consumes embedding throughput, and this is how that throughput is sold.

## The unit

One **credit pack** is **$90** and covers **10 million embedded tokens**. Packs do
not expire. Consumption is metered by the Billing Service, which Ravi Menon owns,
and reported daily to the account owner.

Both directions are metered, but not equally. Indexing a document embeds its full
text once. Answering a question embeds the question — a few dozen tokens — so query
traffic is a rounding error next to ingest for every customer we have. A customer
worried about credit burn should look at their re-ingest schedule before they look
at their query volume.

## What reduces consumption

Three mechanisms, in descending order of effect.

**Checksum skipping.** Re-ingesting an unchanged document costs nothing: the
pipeline compares a content hash before embedding and skips a document that has not
moved. A nightly full re-ingest of a corpus with a 1% change rate therefore costs 1%
of a first run. Customers who rebuild an index from scratch each night are paying
two orders of magnitude more than they need to.

**Late chunking.** The default chunking strategy embeds each document once and pools
the token vectors over chunk spans, rather than embedding each chunk separately. One
forward pass per document replaces one per chunk, so it is cheaper *and* produces
better vectors, because every chunk vector is conditioned on the whole document.

**The embedding cache.** Content-hash keyed, so boilerplate repeated across
documents — a legal footer, a shared header — is embedded once per corpus.

## Compatibility

Metering requires [Retrieval Engine](04-retrieval-engine.md) **2.8 or later**.
Before 2.8 usage was estimated from document counts rather than measured, and
customers still on an older version are billed on that estimate until they upgrade.

## Billing questions

Overage is invoiced monthly in arrears; there is no hard cut-off, because stopping
a customer's ingest mid-run to enforce a credit limit corrupts nothing but wastes
everything already spent on the run. Disputes go to Dana Okafor in Finance. Refunds
follow the same approval thresholds as [expenses](02-expenses-policy.md): above
€500 they need her sign-off.
