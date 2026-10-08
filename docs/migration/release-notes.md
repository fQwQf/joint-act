JointAct migration snapshot: source, prepared Spatial data, and resumable training states.

- Complete canonical Spatial demonstrations: 432 episodes, 52,970 frames; train/val/test = 351/35/46 episodes. Normalization and the 64-mode, 8-step codebook are included.
- JointAct and same-backbone regression checkpoints at steps 100 and 1,000, with optimizer, scheduler, sampler and RNG states.
- The eight-step alignment engineering probe, final five-arm continuation plan, original implementation snapshot, and experiment evidence.
- Archives are split into 512 MiB parts. JSON manifests specify order, size and SHA-256. Restore and base-model verification helpers are included in Git.
- OpenVLA base weights are downloaded from the official pinned revision; they are not duplicated in this release.

Follow `docs/migration/README.md`. Keep the original logical path `/tmp/jointact-1019` as a symlink to persistent storage so checkpoint and study hashes retain their meaning. Rebuild the Python environment on the destination server.

This is a research snapshot, not a claim of improved closed-loop performance. The final five-arm alignment study is prepared but unexecuted.
