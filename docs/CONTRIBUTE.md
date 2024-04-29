# Contribution guidelines
## Git workflow

This project will use the [Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow) for Git. All temporary branches (features, bugs, hotfixes, etc.) must be created and pushed to the "development" branch, the "development" branch will only get merged to main when a stable version is ready. Each release will be tagged as "vMajor.Minor.Patch" according to the [Semantic Versioning](https://semver.org/) definitions. Main will always contain the latest stable version.

### Branch names
Branch names should start with one of the following categories:
- bugfix: For branches working in an important bug fix.
- hotfix: For branches that introduces fast fixes to the codebase.
- feature: For branches that works on a new feature for the current version.
- experimental: When working with experimental stuff, commonly out of the scope of the version but that can be useful in the future.
- docs: For branches that introduces changes in documentation.

The naming convention should include the category, and a brief description of the branch work: _category/brief-branch-description_

for example:
- bugfix/solve-stop-recording-issue
- feature/implement-transcription-queue

### Commits
Commits should be named as [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).