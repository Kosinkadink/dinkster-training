# Security policy

Security fixes target the current main branch. Include the affected version or
commit, operating system, relevant configuration, impact, and minimal
reproduction steps. Do not include credentials, private datasets, or model data.

Report suspected vulnerabilities privately through
[GitHub security advisories](https://github.com/Kosinkadink/dinkster-training/security/advisories/new),
not a public issue. Ordinary bugs belong in this repository's issue tracker.

Training loads user-selected artifacts and runs extension code in worker
processes. Use trusted packs and safe artifact formats. Worker isolation,
filesystem containment, and durable session authorization are security boundaries.
Third-party dependencies have their own licenses and security policies.
