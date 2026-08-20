# Customer Accounts

Account notes for the six documented customers. Products in use, support plan and
account owner live here; ARR, order history and refunds live in the Postgres
`customers`, `products` and `orders` tables, because those change weekly and a
document restating them goes stale without anyone noticing.

This document is the bridge the [Graph Add-on](05-graph-addon.md) traverses: it
connects a customer to a product, and the product documents connect that product to
a service and a person.

## Northwind Traders — United States

Enterprise segment, customer since March 2021. Runs
[Retrieval Engine](04-retrieval-engine.md) in two environments and buys
[Embedding Credits](06-embedding-credits.md) in bulk. On the **Platinum**
[support plan](07-support-plans.md). Account owner **Elena Cruz**.

Their corpus is support tickets, which change constantly, so they re-ingest nightly
and lean hard on checksum skipping. No graph.

## Contoso Ltd — United Kingdom

Enterprise segment, and our first enterprise account — customer since November 2020.
Runs Retrieval Engine plus the **Graph Add-on**; they were the design partner for it.
On the **Platinum** support plan. Account owner **Elena Cruz**.

Contoso is the largest account by ARR. Their quarterly review is with Priya Raman
under the Platinum terms, and they sit in the pre-release channel described in
[Release process](09-release-process.md).

## Fabrikam GmbH — Germany

Mid-market, customer since June 2021. Buys Embedding Credits only; their retrieval
runs on their own stack and they use us for throughput. On the **Gold** support
plan. Account owner **Samir Haddad**.

## Adventure Works — Australia

Mid-market, customer since January 2023. Buys a **Support Plan** and nothing else —
a legacy arrangement from a pilot that was never converted. On the **Silver** tier.
Account owner **Samir Haddad**.

APAC coverage means their business hours fall outside Silver's Lisbon window, which
is the standing reason they are given for upgrading to Gold.

## Tailspin Toys — United States

Small business, customer since August 2023. Bought Embedding Credits once and
refunded the order. On **Bronze** — no paid plan. Account owner **Elena Cruz**.

## Wide World Imports — Vietnam

Small business, customer since February 2024. Runs Retrieval Engine in one
environment. On the **Silver** support plan. Account owner **Samir Haddad**.

## Summary

| Customer | Segment | Support plan | Account owner |
|---|---|---|---|
Northwind Traders | enterprise | Platinum | Elena Cruz |
Contoso Ltd | enterprise | Platinum | Elena Cruz |
Fabrikam GmbH | mid-market | Gold | Samir Haddad |
Adventure Works | mid-market | Silver | Samir Haddad |
Tailspin Toys | small business | Bronze | Elena Cruz |
Wide World Imports | small business | Silver | Samir Haddad |

Elena Cruz covers EMEA and US East; Samir Haddad covers DACH and APAC. Contoso is
the only documented account using the Graph Add-on.
