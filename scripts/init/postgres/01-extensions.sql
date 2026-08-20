-- Extensions and roles. Runs once, on first container start.

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: the vector type + HNSW
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram index for fuzzy ILIKE
CREATE EXTENSION IF NOT EXISTS unaccent;    -- accent-insensitive full-text
CREATE EXTENSION IF NOT EXISTS btree_gin;   -- composite GIN over scalar columns

-- ---------------------------------------------------------------------------
-- Read-only role for Text-to-SQL.
--
-- This is the last line of defence: the SQL guard validates the AST and the
-- executor opens a read-only transaction, but if both are somehow bypassed, a
-- role with no write grants still cannot mutate anything. Defence in depth
-- means the layers fail independently.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ragorc_ro') THEN
    CREATE ROLE ragorc_ro LOGIN PASSWORD 'ragorc_ro';
  END IF;
END
$$;

GRANT CONNECT ON DATABASE ragorc TO ragorc_ro;
GRANT USAGE ON SCHEMA public TO ragorc_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ragorc_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ragorc_ro;

-- Explicitly deny everything else, including future objects.
REVOKE CREATE ON SCHEMA public FROM ragorc_ro;
ALTER ROLE ragorc_ro SET default_transaction_read_only = on;
ALTER ROLE ragorc_ro SET statement_timeout = '15s';
ALTER ROLE ragorc_ro SET idle_in_transaction_session_timeout = '30s';
