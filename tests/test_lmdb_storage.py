"""Storage transaction semantics required by bitmap batch recovery."""
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path

import lmdb
from test_roaring_later_hops import MODULE


class LmdbStorageTests(unittest.TestCase):
    def test_change_index_keeps_latest_revision_and_rolls_back_with_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.lmdb'
            for value in (b'first', b'second', b'third'):
                with MODULE.LmdbStore(path, 'c') as store:
                    store[b'key'] = value
            with MODULE.LmdbStore(path, 'r') as store:
                before = dict(store.transaction.cursor())
                changes = [key for key in before if key.startswith(store._change_prefix)]
                self.assertEqual(len(changes), 1)
            with self.assertRaises(RuntimeError):
                with MODULE.LmdbStore(path, 'w') as store:
                    store[b'key'] = b'rollback'
                    raise RuntimeError('Abort')
            with MODULE.LmdbStore(path, 'r') as store:
                self.assertEqual(dict(store.transaction.cursor()), before)

    def test_writer_waits_for_readers_then_readers_share_writer_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.lmdb'
            with MODULE.LmdbStore(path, 'c') as store:
                store[b'key'] = b'old'
            writer_entered = Event()
            writer_started = Event()

            def write():
                writer_started.set()
                with MODULE.LmdbStore(path, 'w') as writer:
                    writer_entered.set()
                    writer[b'key'] = b'new'
                    with MODULE.LmdbStore(path, 'r') as reader:
                        self.assertIs(reader.environment, writer.environment)
                        self.assertEqual(reader[b'key'], b'old')
                        with self.assertRaises(lmdb.ReadonlyError):
                            reader[b'key'] = b'forbidden'

            with ThreadPoolExecutor(max_workers=1) as pool:
                with MODULE.LmdbStore(path, 'r') as reader:
                    future = pool.submit(write)
                    self.assertTrue(writer_started.wait(2))
                    self.assertFalse(writer_entered.wait(0.02))
                    self.assertEqual(reader[b'key'], b'old')
                future.result(timeout=3)
            with MODULE.LmdbStore(path, 'r') as reader:
                self.assertEqual(reader[b'key'], b'new')

    def test_commit_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.lmdb"
            with MODULE.LmdbStore(path, "c") as store:
                store[b"key"] = b"value"
            with MODULE.LmdbStore(path, "r") as store:
                self.assertEqual(store[b"key"], b"value")
                self.assertIsNone(store.get(b"missing"))
                self.assertNotIn(b"missing", store)
                with self.assertRaises(lmdb.ReadonlyError):
                    store[b"key"] = b"other"

    def test_failure_preserves_checkpoint_and_aborts_later_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.lmdb"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with MODULE.LmdbStore(path, "c") as store:
                    store[b"checkpointed"] = b"saved"
                    store.sync()
                    store[b"unfinished"] = b"not committed"
                    raise RuntimeError("interrupted")
            with MODULE.LmdbStore(path, "r") as store:
                self.assertEqual(store[b"checkpointed"], b"saved")
                self.assertNotIn(b"unfinished", store)


if __name__ == "__main__":
    unittest.main()
