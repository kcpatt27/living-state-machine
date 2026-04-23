# Privacy posture

The Living State Machine stores potentially sensitive personal data (chats,
ingested notes, code). Treat it like a local password manager.

## Defaults

- All state lives on-device.
- Durable memory entries are encrypted at rest with Fernet (AES-128-CBC +
  HMAC-SHA-256) when `LSM_ENCRYPT=true`.
- No telemetry. No outbound calls except to the local Ollama daemon and (when
  you opt in) Weights & Biases.
- `.env` and any key material are `.gitignore`d.

## Key management

Two supported modes:

1. **Passphrase** (recommended for now): you supply `LSM_ENCRYPT_KEY` as a
   Fernet-format base64 key, or run `lsm keys init` to be prompted for a
   passphrase that is stretched via scrypt into a key. The key is held only in
   memory for the life of the process.
2. **DPAPI** (Windows, planned): the key is protected by the current user's
   Windows credentials. No passphrase prompt. Implementation deferred until a
   user asks for it.

If the key is lost, durable memory is unrecoverable. Snapshot before rotating.

## Deletion

`lsm ingest delete --source <path>` removes every entry whose `source` field
matches, including their embeddings and any association/habituation state
keyed on them.

`lsm memory purge --older-than 90d` wipes durable entries older than the cut
while preserving the episodic log (which is treated as append-only audit).
Override with `--include-episodes` if you really mean it.

## Audit

- `memory_store/episodes.sqlite` is append-only; deletions move rows to a
  `tombstones` table rather than hard-deleting, so you can prove what existed.
- Every ingestion creates a provenance record with source path, hash, and
  timestamp.
