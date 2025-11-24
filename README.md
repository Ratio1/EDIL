# Ratio1.ai EDIL (Encrypted Decentralized Inference and Learning) v0.1

EDIL is a research prototype that pairs **latent encoders** with a federated averaging loop. The idea: compress and lightly obfuscate data locally, train models on those embeddings across multiple workers, then aggregate weights centrally. 

The sections below align terminology with Ratio1.ai’s published concepts for R1FS (private IPFS-like storage) and CStore (distributed in-memory state).

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

## Run the simulated end-to-end experiment
1) (Optional) Train the MNIST domain encoder in `edil/experiments/other/ae_test.py` by setting `TRAIN_MODE=True` (stores encoder/decoder in `_cache/`). If you skip this, the demo will try to load pre-existing weights from `_cache/`.
2) Launch the federated simulation: `python edil/experiments/local_test.py`. This loads MNIST, shreds data per worker, trains per round on embeddings, aggregates weights, and reports accuracy.
3) Outputs are printed to stdout; temporary artifacts live in `_cache/`. Use CPU by forcing device in `data_utils.py`/`local_test.py` if you do not have CUDA.

## Extending to Ratio1.ai R1EN network (grounded in Ratio1 blogs)
- **R1FS as storage plane:** Ratio1’s encrypted, sharded, IPFS-like file system. Each file is content-addressed (CID), encrypted, and distributed across edge nodes for redundancy and load balancing. In EDIL, raw data stays on-device; only embeddings or model artifacts are stored/retrieved via R1FS.
- **CStore as coordination/state plane:** a distributed in-memory database (etcd + Redis–like) used to announce and discover CIDs, share lightweight metadata, and synchronize state across nodes. In EDIL, workers publish CIDs for encoder/model artifacts and pull peer updates via CStore hash sets/keys.
- **R1EN as workers:** replace local worker objects with R1EN agents exposing train/test RPCs. Sharding logic is reused; transport shifts to the R1EN network.
- **Data/share flow:** 
  - Store encoded shards or model checkpoints in R1FS → obtain CIDs.
  - Announce CIDs through CStore (hash-set style namespaces) so peers can discover updates.
  - Peers fetch artifacts from R1FS using announced CIDs; state consistency comes from CStore replication.
- **Security & identity:** use mutual TLS and node identity; add attestation where available. R1FS provides content integrity via hashes; encrypt at rest and in transit.
- **Privacy hardening:** add secure aggregation (HE/SMPC) and differential privacy so embeddings/weights are not exposed in plaintext to peers or the aggregator.
- **Operations:** schedule across heterogeneous R1EN hardware, manage stragglers/retries, add telemetry and audit logs. Use CStore versioning/keys to roll out/roll back encoders and checkpoints; monitor R1FS performance (bandwidth/latency) for edge constraints.
- **Lifecycle:** track encoder + classifier versions jointly, propagate to R1ENs, and refresh encoders when accuracy/latency trade-offs drift.

## Current status and cautions
- Works as a single-process simulation only.
- No real encryption, secure aggregation, or networking is present.
- Treat latent encoding as an optimization and weak obfuscation, not as a security guarantee. Real “h-encrypted” training requires cryptographic protections.
