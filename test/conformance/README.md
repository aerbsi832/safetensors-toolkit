# Conformance fixtures

The local conformance corpus is generated from deterministic, tiny containers
so malformed headers and layout boundaries stay reviewable. The official
Python reader check is separate (`scripts/python_interop.sh`) and records its
installed version at runtime; no fixture is described as an upstream oracle
until that command has actually run.
