# Weights

No checkpoint is bundled with the source tree. Publish genuinely trained `.keras` checkpoints as GitHub release assets rather than committing large binaries to Git.

For every released checkpoint, record:

- model variant (`small`, `medium`, or `large`);
- SHA-256 checksum;
- exact resolved configuration;
- training and validation manifest checksums;
- TensorFlow, CUDA, cuDNN, driver, and GPU versions;
- model-selection criterion and selected epoch.

Never publish a dummy or randomly initialized file under a production checkpoint name.
