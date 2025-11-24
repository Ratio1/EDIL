# TODO – Minimal Secure Aggregation Hardening for EDIL (no external HE/SMPC libs)

This plan strengthens the current simulation with additive masking over quantized weights, integrity checks, and freshness. 

>NOTE: This TODOs do not take into account the inherited advanced security features of Ratio1.ai and in particular the R1ENs communications, R1FS or CStore.


## 1) New helper module: `edil/secure_agg.py`
- Constants: `Q = (1 << 61) - 1` (61-bit prime-ish modulus), `SCALE = 10**6` (fixed-point scale).
- Functions:
  - `quantize(arr, S=SCALE) -> np.int64`, `dequantize(arr, S=SCALE) -> np.float64`.
  - `quantize_state(dct, S=SCALE)` – apply quantize to all tensors in a state dict.
  - `mask_state(dct_q, rng, Q=Q)` – sample `mask` ~ Uniform[0, Q) per tensor, return `(cipher, mask)` where `cipher = (q + mask) % Q`.
  - `sum_ciphertexts(states, Q=Q)` – elementwise mod-Q sum of ciphertext state dicts.
  - `unmask_and_dequantize(ct_sum, mask_sums, S=SCALE, Q=Q)` – remove masks and dequantize.
  - Input validation: assert dtypes/shapes match; check `np.max(np.abs(q)) * n_workers < Q/4` to warn on overflow risk.

## 2) Update `th_utils.weights_getter` / `weights_loader`
- Add `secure=False, scale=SCALE` parameters.
- If `secure=True`, return `np.int64` quantized weights via `quantize_state`.
- `weights_loader` must accept both float32 and int64; if int64, dequantize before loading into torch tensors.

## 3) Add secure path in aggregation (`th_utils.aggregate_function`)
- Signature: `aggregate_function(..., secure=False, mask_sums=None, Q=Q, scale=SCALE)`.
- If `secure`:
  - Expect `workers` entries to be ciphertext dicts (int64, mod Q).
  - Compute `ct_sum = sum_ciphertexts(workers)`.
  - Require `mask_sums` (dict of summed negative masks from workers, mod Q).
  - `plain = unmask_and_dequantize(ct_sum, mask_sums, scale, Q)`.
  - Load `plain` into destination model via `weights_loader`.
- Else: keep existing float FedAvg.

## 4) Worker-side flow (`SimpleProcessingNode.local_train` return path)
- After training:
  - Get float state via `model_weights_getter`.
  - Quantize: `q_state = quantize_state(state)`.
  - Mask: `ct_state, mask_state = mask_state(q_state, rng)`.
  - Keep `mask_state` locally to send `(-mask) % Q` later.
  - Build metadata: `round_id, worker_id, nonce, hash(ct_state)`; sign with HMAC (see below).
  - Return `(ct_state, metadata)` instead of raw weights when `secure=True`.

## 5) Coordinator flow (`SimpleProcessingNode.distributed_train`)
- Add `secure=False` flag.
- If `secure`:
  - Collect ciphertexts from workers; store per-worker masks (or instruct workers to later send `-mask`).
  - Before aggregation, gather `mask_sums = sum_over_workers((-mask) % Q)` elementwise.
  - Pass `secure=True` and `mask_sums` into `aggregate_function`.
  - Validate signatures, round IDs, nonces; reject replayed or mismatched payloads.
- If not secure: existing behavior.

## 6) Integrity & freshness (no new libs)
- Use `hmac` + `hashlib.sha256` with a shared secret to sign `(round_id, worker_id, nonce, hash(ciphertext_bytes))`.
- Include `nonce` (random 128-bit) and optionally a timestamp; store last-seen nonce per worker to reject replays.
- Hashing helper: serialize state dict deterministically (e.g., concatenate `key||bytes(value.tobytes())`).

## 7) Parameterization and safety checks
- Configurable `Q` and `SCALE`; add preflight: `max_abs * SCALE * n_workers < Q/4`.
- Add a `secure_rng` seeded per worker/process.
- Add warnings if overflow risk detected.

## 8) Testing scaffolding
- Unit test (toy): 2–3 tensors with known floats; quantize → mask → sum → unmask; expect dequantized sum matches plain sum within tolerance.
- Integration test: 2-worker training loop using secure path; compare aggregated model (secure) vs plain FedAvg on tiny data.
- HMAC test: tamper with ciphertext; expect verification failure.

## 9) Documentation notes (in README/ANALYSIS)
- Clarify that secure mode uses additive masking (not semantic HE, not SMPC).
- State limitations: no protection against colluding aggregator + worker; only sums supported; relies on shared secrets and correct mask handling.

## 10) Known limitations to call out
- No semantic security; masks plus ciphertext reveal plaintext.
- No ciphertext multiplication; only additive aggregation.
- Key management, mTLS, and attestation are still required externally.
- Collusion between aggregator and any worker breaks confidentiality of others.

These steps give a minimal “secure aggregation–like” path without external libraries and make the prototype harder to misuse as plaintext FL. They are not a substitute for real HE/SMPC/DP.***
