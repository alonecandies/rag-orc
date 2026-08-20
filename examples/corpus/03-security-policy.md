# Security Policy

Owned by Marcus Vale, Security Lead. Applies to all systems that touch customer
data, which includes every service listed in [On-call](08-engineering-oncall.md).

## Access control

- Single sign-on is mandatory. There are no local passwords on any internal system.
- Production access requires a **hardware security key**. Software TOTP is accepted
  for internal tools and never for production.
- Access is reviewed **quarterly**. An entitlement nobody has used in 90 days is
  revoked automatically; asking for it back takes one ticket and is preferable to
  an account with standing access nobody remembers granting.
- Secrets live in Vault. A credential in a repository, a ticket or a chat message is
  treated as compromised the moment it is discovered, and rotated the same day.

Customer data may not be copied to a laptop. Debugging happens against synthetic
fixtures or in the production shell, whose sessions are recorded.

## Incident severities

These four definitions are the shared vocabulary for on-call escalation and for the
response commitments in [Support Plans](07-support-plans.md).

| Severity | Definition |
|---|---|
**SEV-1** | Customer data is exposed, lost, or the platform is wholly unavailable. |
**SEV-2** | A core capability is unavailable or badly degraded for one or more customers, with no workaround. |
**SEV-3** | A capability is degraded and a workaround exists. |
**SEV-4** | A defect with no customer impact, or a question. |

Severity is set by the responder and can only be *raised* by the responder;
lowering a severity requires the security lead or the service owner, so that
nobody can quietly downgrade an incident they are being paged for.

## Reporting

Report anything suspicious to Marcus Vale directly, in the `#security` channel, or
through the anonymous form on the intranet. There is no penalty for a false
positive, and there is one for sitting on a real one.

A SEV-1 opens an incident channel, pages the on-call primary and notifies Priya
Raman within 15 minutes. Customer notification for a data-exposure SEV-1 goes out
within 72 hours and is written by Marcus Vale with Yuki Tanabe.

## Retrieval-specific rules

Retrieved documents are untrusted input. Any pipeline that puts customer content
into a model prompt must run it through the injection scanner and structural
isolation first. Generated SQL and Cypher execute only against read-only roles,
never against the application role.
