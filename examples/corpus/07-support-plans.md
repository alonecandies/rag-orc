# Support Plans

Owned by Yuki Tanabe, Support Lead. Four tiers. The response commitments below are
stated in terms of the severity definitions in
[Security](03-security-policy.md) — there is deliberately only one severity scale in
the company, so an engineer's SEV-2 and a customer's SEV-2 are the same thing.

## The tiers

| Plan | Price | SEV-1 response | SEV-2 response | Hours |
|---|---|---|---|---|
**Bronze** | included | next business day | next business day | business hours, Lisbon |
**Silver** | $300 / month | 8 business hours | 2 business days | business hours, Lisbon |
**Gold** | per contract | 2 hours | 8 business hours | 24×5 |
**Platinum** | per contract | 30 minutes | 4 hours | 24×7 |

"Response" means a named engineer has acknowledged the ticket and started work. It
is not a resolution commitment, and we do not sell one: a fix time we cannot control
is a promise we would break.

## What each tier adds

- **Bronze** — ticket support, documentation, the public status page. Every
  Retrieval Engine customer has it whether they buy anything or not.
- **Silver** — the paid entry tier. Adds a shared support channel and quarterly
  usage reviews with the account owner.
- **Gold** — adds 24×5 coverage, a named support engineer, and the ability to raise
  the sustained query-rate limit described in
  [Retrieval Engine](04-retrieval-engine.md).
- **Platinum** — adds 24×7 coverage, a quarterly review with Priya Raman, and
  inclusion in the pre-release channel described in
  [Release process](09-release-process.md).

## Escalation

A customer cannot page the engineering on-call rotation directly. Support triages,
sets the severity, and pages through the ladder in
[On-call](08-engineering-oncall.md) when the severity warrants it. That indirection
exists because severity is a judgement about *impact across all customers*, and the
reporter of an incident is the person least able to make it.

Platinum customers may request a specific engineer; whether they get one depends on
the rotation, and the request never changes the response clock.

## Which customers are on which plan

Per-account assignments are in [Customer accounts](10-customer-accounts.md). Two of
the six documented accounts are on Platinum, and both are enterprise segment — the
tier is priced per contract precisely so that it can be sized to an account rather
than to a list.
