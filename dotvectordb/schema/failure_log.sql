CREATE TABLE IF NOT EXISTS failure_log (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase TEXT NOT NULL CHECK (
        phase IN ('chunk', 'entity', 'relation', 'desc')
    ),
    item_type TEXT NOT NULL CHECK (
        item_type IN ('chunk', 'entity', 'relation')
    ),
    item_id TEXT NOT NULL,
    error_message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS failure_log_phase_item_index
ON failure_log (phase, item_type, item_id);
