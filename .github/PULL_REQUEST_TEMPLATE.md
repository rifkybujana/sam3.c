## Description

Please include a summary of the change and which issue is fixed.

Please also include relevant motivation and context. Describe your key design decisions and how they fit into the larger architecture.

Fixes # (issue)

## Type of Change

* [ ] Bug fix (non-breaking change which fixes an issue)
* [ ] New feature (non-breaking change which adds functionality)
* [ ] Breaking change (fix or feature that would cause existing functionality/APIs to change)
* [ ] Refactor (improving codebase without changing external behavior)
* [ ] Documentation update

## Developer Checklist

All PRs must adhere strictly to the guidelines in `CONTRIBUTING.md` and `CLAUDE.md`.

* [ ] **Compilation**: It compiles cleanly with `-std=c11 -Wall -Wextra -Wpedantic` using GCC/Clang.
* [ ] **Indentation**: I have used tabs (8 characters wide) for indentation.
* [ ] **Line Length**: I have kept lines within an 80-column soft limit and 100-column hard limit.
* [ ] **Braces**: I used the K&R brace style for functions (opening brace on its own line).
* [ ] **Naming**: I have used `snake_case` for everything and prefixed public API symbols with `sam3_`.
* [ ] **C11 Only**: There are absolutely no C++ syntax or C++ features in the code.
* [ ] **Memory**: I have verified allocations are safely managed, and have avoided raw `malloc`/`free` in hot-paths utilizing the arena allocator where appropriate.
* [ ] **Dependencies**: I have not included any additional dependencies without discussion.
* [ ] **Documentation Header**: New files start with the required standard C-style comment block header (description, key types, dependencies, copyright).
* [ ] **Tests**: I have added tests to `tests/test_<module>.c` that prove my fix is effective or my feature works.
* [ ] **CTest Passing**: I have run the test suite locally (`make && ctest --output-on-failure`) and all tests pass (including ASan/UBSan locally).