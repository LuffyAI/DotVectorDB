# DotVectorDB

## Pinned Vec1 version

Source builds compile SQLite's **Vec1 0.8** from the vendored source at
`dotvectordb/compiled_vec1/original_files/ext/vec1/vec1.c`.
The downloaded SQLite source revision is
`213d76c009bcd860b5a624be6ffe3b89aab063bc6f2c30bdb40384c5d7f8b35f`.
This is the Vec1 extension version, separate from the SQLite engine version.
The SHA-256 of the pinned `vec1.c` is
`b341c15fe10e41a9b96981ff3a9c217ea0774b4e04932b5acf3dac1320aeceb1`.

## Installation and native compilation

Automatic compilation currently supports Linux. Install `make` and a C compiler
first (on Debian/Ubuntu: `sudo apt install build-essential`). Matching SQLite
headers are bundled in `dotvectordb/compiled_vec1/headers`; `sqlite3.h` was generated with the
downloaded source's `tool/mksqlite3h.tcl`, and `sqlite3ext.h` is copied unchanged
from its `src/` directory. The extension uses Python's SQLite engine at runtime.

### Bitmap storage dependency

Roaring bitmap storage uses **LMDB** through the pip-installable `lmdb` package.
It is declared in `requirements.txt` and the package's `install_requires`, so
`pip install .` installs it automatically. Scripts and notebook kernels use
exactly the same storage implementation; no system Python bindings or path
workarounds are needed.

The index contains `entity_ids.lmdb` and `bitmaps.lmdb`, with LMDB-managed
`-lock` files. Stores use transactional writes and workers open their own
read-only environments after starting. The 1 TiB map-size limit reserves
virtual address space, not 1 TiB of RAM or preallocated file contents.

Existing GNU dbm bitmap files are a different storage format; this change does
not convert them. New bitmap indexes use the `.lmdb` paths.

From this source checkout:

```sh
python -m pip install .
```

The package build invokes `dotvectordb/compiled_vec1/Makefile` and compiles a fresh Vec1 0.8 library
into the build's `dotvectordb/compiled_vec1/vec1.so`. Pip installs it in that
same location inside the installed package. Existing `vec1.so` (0.6) and
`vec1_gl.so` (0.7) binaries in the checkout are excluded from distributions.
No network download of Vec1 is needed during compilation.

The default compiler flags optimize for the build machine's CPU
(`-march=native`). Do not distribute those wheels to other machines.
Override `CC`, `CPPFLAGS`, `CFLAGS`, or `LDFLAGS` as needed. To compile
directly into the checkout, run `make -C dotvectordb/compiled_vec1`.

For users to compile locally from PyPI, publish the source distribution
(`.tar.gz`). Prebuilt wheels skip compilation, and pip can reuse cached
wheels. To explicitly request a fresh source build:

```sh
python -m pip install --no-binary=dotvectordb --no-cache-dir dotvectordb
```

After loading the installed extension, `SELECT vec1_info();` should report
`version 0.8` with the enabled SIMD and threading options.
