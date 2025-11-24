# Ratio1.ai EDIL (Encrypted Decentralized Inference and Learning)

EDIL is a research prototype that pairs **latent encoders** with a federated averaging loop. The idea: compress and lightly obfuscate data locally, train models on those embeddings across multiple workers, then aggregate weights centrally. It is a simulation of on-edge training; there is no real cryptography or networking yet.

## Idea: latent encoding for FL
- **What it is:** A domain autoencoder turns raw inputs (e.g., images) into compact embeddings. Training and aggregation happen on these embeddings instead of raw data.
- **Potential benefits:** 
  - Cuts bandwidth and speeds up local training.
  - Provides mild obfuscation vs. raw data leakage.
  - Allows reuse of a domain encoder across many tasks.
- **Limitations:** 
  - Embeddings are not encryption; a decoder or adversary can often reconstruct inputs.
  - No semantic security, key management, secure aggregation, or DP by default.
  - Workers and the aggregator see plaintext embeddings and weights.

## How the current simulation works
1) **Train or load a domain encoder** (`SimpleDomainAutoEncoder`) on local data to produce embeddings.
2) **Shard data across workers** using `sample_shards`, respecting worker load percentages.
3) **Local training per worker:** each worker receives initial weights, trains with `SimpleTrainer`, evaluates with `SimpleTester`, and returns updated weights.
4) **Aggregate models:** coordinator averages numpy-formatted state dicts (`aggregate_function`) and repeats for multiple rounds.
5) **Evaluate end-to-end:** combine the domain encoder with the aggregated classifier for inference.

## Development environment
- Prefer the `.devcontainer/` setup for reproducible CUDA/PyTorch tooling (`devcontainer.json`, `Dockerfile`). Launching the repo in a devcontainer pulls the right dependencies without touching the host.
- If you run locally, mirror the dependencies in `edil/experiments`, but expect CUDA defaults in the scripts; switch to CPU in `data_utils.py` / `local_test.py` if needed.

## Extending to Ratio1.ai R1EN network
- **R1EN as workers:** replace the local worker objects with R1EN agents that expose train/test RPCs. Sharding stays the same; transport becomes networked.
- **CStore as artifact backbone:** store/version domain encoders and aggregated model checkpoints in CStore; workers pull encoder versions from CStore and push signed weight updates back for aggregation.
- **R1FS as data plane:** mount or stream local datasets through R1FS so that only embeddings leave the device. Keep raw data on-device; enforce access control via R1FS policies.
- **Secure transport & identity:** require mutual TLS and node identity for R1ENs; add attestation if available so only trusted nodes participate.
- **Privacy hardening:** add secure aggregation (HE/SMPC) and optional differential privacy so embeddings/weights are never exposed in plaintext to peers or the aggregator.
- **Operations:** add scheduling for heterogeneous R1EN hardware, straggler handling, retries, telemetry, and audit logs. Automate encoder rollout/rollback via CStore versioning.
- **Model lifecycle:** track encoder + classifier versions jointly, propagate updates to R1ENs, and monitor accuracy vs. bandwidth/latency to decide when to refresh encoders.

## Current status and cautions
- Works as a single-process simulation only.
- No real encryption, secure aggregation, or networking is present.
- Treat latent encoding as an optimization and weak obfuscation, not as a security guarantee. Real “h-encrypted” training requires cryptographic protections.
