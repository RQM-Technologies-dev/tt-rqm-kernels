# H2A clean-commit reproduction

This retained package binds the compensated H2A candidate to implementation
commit `225cb213ae79df7acd43d6056841c3eae7b5fc40`. Two isolated clean builds
produced the same candidate SHA-256 and ELF build ID. The compensated primitive
probe, axis smoke, mixed zero/nonzero case, large-angle case, and complete
nine-case suite passed on N300 device 0 with zero failing and zero nonfinite
values.

Every clean-run output checksum exactly matches the earlier dirty-tree
compensated pilot. The identity changed because the source is now clean and
committed; the numerical outputs did not.

This remains development evidence: `designated=false`,
`qualification_eligible=false`, `claim_level=null`, `stable_benchmark=false`,
and `performance_eligible=false`. It is not a Claim Level 0 session.
