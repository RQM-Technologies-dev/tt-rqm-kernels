# H2B clean-build reproduction

Two fresh builds from source commit `796e7459c61a81196da1cafd51eb78af6bc9fc27`, source bundle `57a6f741dae34d4d029cbd4274fd59cb5e4e2d34a2617e841d4bbb95d4486d01`, and TT-Metal commit `dd2849b5bc6b7a5d38a9eafbeba31ef8d530f8d4` produced byte-identical executables.

- Candidate SHA-256: `3f7069bd1376a4081600a77c23ac8bb0a5b4273167d98ba20117581b9def7913`
- Compiler path class: GNU C++ 11.4.0 through Open MPI 5.0.7 with ULFM
- Source trees: clean for both builds

An earlier environment preflight used the system-default MPI library, which lacked required ULFM symbols and failed at link time. It produced no candidate binary, executed no candidate, and did not consume a pilot attempt.
