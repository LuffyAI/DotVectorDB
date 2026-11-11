# Vec1 0.8, SQLite revision recorded in README.MD.
VEC1_SOURCE = original_files/ext/vec1/vec1.c
OUTPUT_DIR ?= .
CFLAGS ?= -O3 -DNDEBUG -march=native

# Always rebuild: an existing binary may belong to a different machine/version.
.PHONY: vec1
vec1:
	mkdir -p "$(OUTPUT_DIR)"
	$(CC) -I headers $(CPPFLAGS) $(CFLAGS) -fPIC -shared -pthread "$(VEC1_SOURCE)" -o "$(OUTPUT_DIR)/vec1.so" $(LDFLAGS) -lm
