# Halcyon Data — Company Handbook

Halcyon Data is a fictional company. This corpus exists so the examples in
`examples/` have something real to retrieve, traverse and be graded against.

## Who we are

Halcyon Data was founded in 2019 in Lisbon and builds retrieval infrastructure for
teams with large private document sets. We are 140 people across three offices —
Lisbon (engineering and support), London (sales) and Austin (customer success) —
and closed a Series B in 2023.

We sell four things. The platform products are **Retrieval Engine** and the
**Graph Add-on**; usage is billed through **Embedding Credits**; and human help is
sold as a **Support Plan**. Each has its own product document:
[Retrieval Engine](04-retrieval-engine.md), [Graph Add-on](05-graph-addon.md),
[Embedding Credits](06-embedding-credits.md), [Support Plans](07-support-plans.md).

## Who owns what

| Area | Owner | Role |
|---|---|---|
Engineering | Priya Raman | VP Engineering |
Graph Add-on | Tomas Lindqvist | Staff Engineer |
Index Service | Bea Nkemelu | Site Reliability Engineer |
Billing Service | Ravi Menon | Senior Engineer |
Security | Marcus Vale | Security Lead |
Finance | Dana Okafor | Head of Finance |
Support | Yuki Tanabe | Support Lead |
Accounts (EMEA + US East) | Elena Cruz | Account Manager |
Accounts (DACH + APAC) | Samir Haddad | Account Manager |

Priya Raman is the escalation point for every engineering service; see
[On-call](08-engineering-oncall.md) for the rotation and the escalation ladder.

## Policies every employee needs

- [Expenses and travel](02-expenses-policy.md) — approval thresholds and per-diems.
  Dana Okafor approves anything above the manager threshold.
- [Security](03-security-policy.md) — access control, the incident severity
  definitions, and how to report something.
- [Release process](09-release-process.md) — version numbering and the release
  train that every product ships on.

## Customers

Six accounts are documented in [Customer accounts](10-customer-accounts.md), with
the segment, the products in use and the support plan for each. Commercial figures
— ARR, order history, refunds — live in the `customers`, `products` and `orders`
tables in Postgres rather than in this handbook, because they change weekly and a
document that restates them goes stale silently.
