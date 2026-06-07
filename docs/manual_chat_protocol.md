# Manual Chat Protocol (5-10 Minute Smoke Test)

This protocol allows you to manually verify the facts-backed pipeline, episodic memory retrieval, and slot-attribution dynamics in an interactive `lsm chat` session. It mirrors the shape of the `recall_pref_v2` benchmark but uses a real, live session to ensure product-level parity.

**Automated smoke (no Ollama):** `uv run pytest tests/test_manual_chat_protocol.py -q`

**Hybrid recurrent core in live chat (opt-in):** set `LSM_STATE_UPDATER=hybrid` (and ensure `runs/small/minimal/ckpts/step_002500.pt` exists) before `uv run lsm chat`.

---

## Step 1: Clean Slate

Before starting, prune any stale facts and memory entries to ensure a clean test environment:

```powershell
# Prune all extracted facts
uv run lsm fact prune --all --yes

# (Optional) Purge older memory entries
uv run lsm memory purge --older-than 0d
```

---

## Step 2: Assert Preference (Turn 1)

Start the interactive chat REPL:

```powershell
uv run lsm chat
```

Once the chat loads, assert your favorite drink (phrasing the extractor accepts today):

```
user> just for context: I prefer negroni. please remember that.
```

**Expected Assistant Response:**
- The assistant should acknowledge the preference (e.g., "Got it, I will remember that your favorite drink is negroni.").

---

## Step 3: Out-of-Band Verification

Keep the chat session running. Open a **second terminal** in the same directory and inspect the state:

```powershell
# 1. Check extracted facts
uv run lsm fact list

# 2. Check current state projection s_t
uv run lsm state show
```

**Expected Output:**
- `lsm fact list` should show a fact under `user.preferences` with the value `negroni`.
- `lsm state show` should show that `negroni` is successfully projected into the state text.

---

## Step 4: Distraction & Decoy Turns (Turns 2-4)

Go back to the chat terminal and feed it some distraction and decoy turns:

```
user> what do you think about the weather today?
assistant> [Acknowledges weather...]

user> can you tell me a short joke?
assistant> [Tells a joke...]

user> random note: someone at work loves hibiscus tea, not that it matters.
assistant> [Acknowledges coworker's drink...]
```

---

## Step 5: Decoy Verification

In your **second terminal**, re-run the state checks:

```powershell
uv run lsm fact list
uv run lsm state show
```

**Expected Output:**
- `hibiscus tea` should **not** be attributed to you. The `user.preferences` slot for favorite drink must still be `negroni`.
- The coworker's preference may be written to episodic memory, but it should not pollute your structured facts or state projection.

---

## Step 6: Recall Question (Turn 5)

Go back to the chat terminal and ask the recall question:

```
user> what is my favorite drink?
```

**Expected Assistant Response:**
- The assistant should answer "negroni" (or "Your favorite drink is negroni.").

---

## Step 7: Memory Query Verification

Exit the chat REPL (`exit` or `Ctrl+C`). Run a manual memory query to verify episodic retrieval:

```powershell
uv run lsm memory query "negroni"
```

**Expected Output:**
- You should see scored hits from the conversation history containing the preference turn.

_Note: with production `fastembed`, `uv run lsm memory query "favorite drink"` also works semantically; hash-style tests use the literal token instead._
