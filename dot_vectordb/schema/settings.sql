/*********************************
    SETTINGS
*********************************/
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 10000;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 30000000000;
PRAGMA cache_size = -200000;
PRAGMA foreign_keys = ON;