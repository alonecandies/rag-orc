-- A small business schema so Text-to-SQL has something real to query, and the
-- routing layer has a reason to pick `relational` over `vector`.

CREATE TABLE IF NOT EXISTS customers (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    country      TEXT NOT NULL,
    segment      TEXT NOT NULL CHECK (segment IN ('enterprise', 'mid_market', 'smb')),
    signed_up_at DATE NOT NULL,
    arr_usd      NUMERIC(12, 2) NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS products (
    id       BIGSERIAL PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,
    price_usd NUMERIC(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id          BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers (id),
    product_id  BIGINT NOT NULL REFERENCES products (id),
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    total_usd   NUMERIC(12, 2) NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'refunded')),
    ordered_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS orders_customer_idx ON orders (customer_id);
CREATE INDEX IF NOT EXISTS orders_ordered_at_idx ON orders (ordered_at DESC);
CREATE INDEX IF NOT EXISTS customers_country_idx ON customers (country);

INSERT INTO customers (name, country, segment, signed_up_at, arr_usd) VALUES
  ('Northwind Traders', 'US', 'enterprise', '2021-03-14', 480000),
  ('Contoso Ltd',       'GB', 'enterprise', '2020-11-02', 620000),
  ('Fabrikam GmbH',     'DE', 'mid_market', '2022-06-21', 145000),
  ('Adventure Works',   'AU', 'mid_market', '2023-01-09',  98000),
  ('Tailspin Toys',     'US', 'smb',        '2023-08-30',  24000),
  ('Wide World Imports','VN', 'smb',        '2024-02-17',  31000)
ON CONFLICT DO NOTHING;

INSERT INTO products (name, category, price_usd) VALUES
  ('Retrieval Engine',   'platform',  1200.00),
  ('Graph Add-on',       'platform',   450.00),
  ('Embedding Credits',  'usage',       90.00),
  ('Support Plan',       'services',   300.00)
ON CONFLICT DO NOTHING;

INSERT INTO orders (customer_id, product_id, quantity, total_usd, status, ordered_at) VALUES
  (1, 1, 2, 2400.00, 'delivered', '2024-01-15T10:00:00Z'),
  (1, 3, 10, 900.00, 'delivered', '2024-02-01T10:00:00Z'),
  (2, 1, 5, 6000.00, 'delivered', '2024-01-20T10:00:00Z'),
  (2, 2, 1,  450.00, 'shipped',   '2024-03-11T10:00:00Z'),
  (3, 3, 20, 1800.00,'delivered', '2024-02-14T10:00:00Z'),
  (4, 4, 1,  300.00, 'pending',   '2024-04-02T10:00:00Z'),
  (5, 3, 3,  270.00, 'refunded',  '2024-03-28T10:00:00Z'),
  (6, 1, 1, 1200.00, 'delivered', '2024-04-19T10:00:00Z')
ON CONFLICT DO NOTHING;

GRANT SELECT ON customers, products, orders TO ragorc_ro;
