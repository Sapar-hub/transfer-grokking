"""Post-download setup for Phi-2 HF cache.

After curl finishes downloading model-00001-of-00002.safetensors,
this script verifies the hash and creates the proper symlinks.
"""

import hashlib
import os
import sys

BLOBS_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--phi-2/blobs")
SNAPSHOTS_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--phi-2/snapshots")
SNAPSHOT = "810d367871c1d460086d9f82db8696f2e0a0fcd0"
EXPECTED_HASH = "7fbcdefa72edf7527bf5da40535b57d9f5bd3d16829b94a9d25d2b457df62e84"
FILENAME = "model-00001-of-00002.safetensors"

blob_path = os.path.join(BLOBS_DIR, EXPECTED_HASH)
snapshot_path = os.path.join(SNAPSHOTS_DIR, SNAPSHOT, FILENAME)

if not os.path.exists(blob_path):
    print(f"Blob not found: {blob_path}")
    print("Download not complete yet.")
    sys.exit(1)

size = os.path.getsize(blob_path)
print(f"Blob size: {size} bytes ({size/1e9:.2f} GB)")

# Quick check: is it at least close to expected size?
expected_size = 4995584424  # from x-linked-size header
if size < expected_size:
    print(f"Download incomplete: {size} < {expected_size} ({size/expected_size*100:.1f}%)")
    sys.exit(1)

# Verify SHA256
print("Verifying SHA256...")
h = hashlib.sha256()
with open(blob_path, 'rb') as f:
    while True:
        chunk = f.read(65536)
        if not chunk:
            break
        h.update(chunk)

computed = h.hexdigest()
print(f"Expected: {EXPECTED_HASH}")
print(f"Computed: {computed}")

if computed != EXPECTED_HASH:
    print("Hash MISMATCH! Removing corrupt file...")
    os.remove(blob_path)
    print("Removed. Re-download required.")
    sys.exit(1)

print("Hash MATCH.")

# Create symlink
if os.path.exists(snapshot_path):
    print(f"Symlink already exists: {snapshot_path}")
else:
    rel_path = os.path.relpath(blob_path, os.path.dirname(snapshot_path))
    os.symlink(rel_path, snapshot_path)
    print(f"Created symlink: {snapshot_path} -> {rel_path}")

print("Phi-2 cache is ready.")
print("You can now run: python random_baseline.py")
