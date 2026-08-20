# Engineering On-call

One weekly rotation per service, primary and secondary. Handover is Tuesday at
10:00 Lisbon time, which is the same slot as the release train in
[Release process](09-release-process.md) — deliberately, so the person who takes the
pager is the person who watched the deploy.

## Services and owners

| Service | What it does | Owner and on-call primary |
|---|---|---|
**Query Gateway** | terminates queries, fans out to retrieval, streams answers | Priya Raman |
**Index Service** | ingest, splitting, embedding, writes to the vector store | Bea Nkemelu |
**Graph Service** | entity graph, community summaries, traversals for the [Graph Add-on](05-graph-addon.md) | Tomas Lindqvist |
**Billing Service** | usage metering for [Embedding Credits](06-embedding-credits.md) | Ravi Menon |

The service owner is the default on-call primary for their own service. The
secondary is drawn from the other three teams on rotation, so every engineer sees
every service at least once a quarter and no service has a single point of
knowledge.

## The escalation ladder

1. **On-call primary** for the affected service — paged automatically.
2. **On-call secondary** — paged after 10 minutes without acknowledgement.
3. **Service owner**, if they are not already the primary.
4. **Priya Raman**, VP Engineering — the escalation point for every service.
5. **CTO**, for a SEV-1 that is still open after two hours.

A SEV-1 involving customer data also notifies Marcus Vale immediately, in parallel
with step 1 rather than after it, because the security response and the availability
response are different work happening at the same time. Severity definitions are in
[Security](03-security-policy.md).

## What the primary owes

- Acknowledge inside 10 minutes during their week.
- Keep the incident channel updated every 30 minutes on a SEV-1 and every 2 hours on
  a SEV-2, even when the update is "no change" — silence is indistinguishable from
  nobody looking.
- Write the incident review within three business days. Reviews are blameless and
  are read at the Thursday engineering meeting.

## Degradation expectations

A Graph Service outage degrades queries to passage ranking and is therefore a
SEV-2. An Index Service outage stops ingest but leaves queries working, also a
SEV-2. A Query Gateway outage is a SEV-1: nothing answers. Billing Service downtime
is a SEV-3 as long as it is under four hours, because metering buffers and replays;
past four hours the buffer is at risk and it becomes a SEV-2.
