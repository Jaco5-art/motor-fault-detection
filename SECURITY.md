# Security policy

## Model artifact safety

The bundled GMM is serialized with `joblib`, which can execute code during deserialization. The CLI loads only the repository-owned artifact under `src/motor_fault/resources/gmm/` by default. Never use `--bundle-dir` with files from an untrusted source.

## Reporting

Do not publish raw competition data, local paths, credentials, access tokens, or proprietary machine telemetry in an issue. Report security-sensitive problems privately to the repository owner after publication.
