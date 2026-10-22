CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,

    role TEXT NOT NULL
        CHECK (role IN ('administrator', 'developer', 'user')),

    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1))
);


CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,

    kind TEXT NOT NULL
        CHECK (kind IN ('personal', 'shared', 'test')),

    owner_id TEXT REFERENCES users(id),

    CHECK (
        (kind = 'personal' AND owner_id IS NOT NULL)
        OR
        (kind IN ('shared', 'test') AND owner_id IS NULL)
    )
);


CREATE TABLE IF NOT EXISTS collection_permissions (
    user_id TEXT NOT NULL REFERENCES users(id),
    collection_id TEXT NOT NULL REFERENCES collections(id),

    access_level TEXT NOT NULL
        CHECK (access_level IN ('read', 'write', 'manage')),

    PRIMARY KEY (user_id, collection_id)
);


CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id
ON sessions(user_id);


CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id),
    created_by TEXT NOT NULL REFERENCES users(id),

    original_name TEXT NOT NULL,
    storage_name TEXT NOT NULL UNIQUE,

    extension TEXT NOT NULL
        CHECK (extension IN ('.txt', '.docx', '.pdf')),

    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    sha256 TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'uploaded'
        CHECK (status IN ('uploaded', 'processing', 'ready', 'failed')),

    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_collection_id
ON documents(collection_id);

CREATE INDEX IF NOT EXISTS idx_documents_collection_sha256
ON documents(collection_id, sha256);