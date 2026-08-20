# Release Process

Every Halcyon Data product ships on one train. There are no per-product release
calendars, because two products that must be compatible cannot have independent
schedules without someone tracking the matrix by hand.

## The train

Releases go out **every second Tuesday at 10:00 Lisbon time**. Handover of the
on-call pager happens in the same slot — see
[On-call](08-engineering-oncall.md) — so the engineer holding the pager for the week
is the one who watched the deploy go out.

A change misses the train rather than delaying it. There are two exceptions: a
SEV-1 fix, and a security patch signed off by Marcus Vale. Both ship immediately and
are announced on the status page.

## Version numbers

Semantic versioning, and the middle number carries a commitment.

- **Major** — an index rebuild or a credential change is required. We have done this
  once, at 3.0.
- **Minor** — new capability, backwards compatible. The current release is
  [Retrieval Engine](04-retrieval-engine.md) **3.4**.
- **Patch** — fixes only.

A compatibility floor is always stated as a minor version, never as a patch: the
[Graph Add-on](05-graph-addon.md) requires 3.0 or later and
[Embedding Credits](06-embedding-credits.md) metering requires 2.8 or later. Stating
a floor at patch granularity would make every patch a potential compatibility
event.

## Environments

Changes land in staging on the Thursday before the train and soak for four days.
Customers on the Platinum [support plan](07-support-plans.md) have access to the
pre-release channel and see the staging build during the soak, which is where most
of the compatibility reports we act on come from.

## Deprecation

A capability is announced as deprecated at a minor release, keeps working for two
further minor releases, and is removed at the third. That is roughly three months at
the current cadence. Removals are never patched in.

## Who can release

The service owner for the affected service, or Priya Raman. The release runbook is
in the engineering wiki; it requires a green staging soak, a signed-off change log,
and an acknowledged pager.
