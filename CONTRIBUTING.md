# Contributing to datia-synth

Thank you for your interest in contributing to datia-synth!

## How to contribute

### Reporting bugs

Open an issue at https://github.com/iantum/datia-synth/issues with:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS

### Suggesting features

Open an issue with the `enhancement` label describing:
- The use case
- Why it fits the project's scope (mobility/health synthetic data)

### Submitting code

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests if applicable
5. Run the test suite: `pytest`
6. Submit a pull request

## Development setup

```bash
git clone https://github.com/iantum/datia-synth.git
cd datia-synth
pip install -e ".[all]"
```

## Code style

- Follow PEP 8
- Use type hints where practical
- Keep functions focused and documented

## Scope

This library is developed in the context of Proyecto DatIA (mobility and health data for vulnerable communities in the Comunitat Valenciana). Contributions should align with this scientific and social mission.

## Licence

By contributing, you agree your contributions will be licensed under Apache 2.0.
