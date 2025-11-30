# Publishing Palace to PyPI

This guide explains how to publish Palace to PyPI (Python Package Index).

## Prerequisites

1. **PyPI Account**: Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

2. **API Tokens**: Generate API tokens for secure authentication:
   - Go to Account Settings → API tokens
   - Create a token with upload permissions
   - Save it securely (you won't see it again)

3. **Install Publishing Tools**:
   ```bash
   uv pip install twine
   ```

## Building the Package

Build both source distribution and wheel:

```bash
# Clean previous builds
rm -rf dist/

# Build using uv (recommended)
uv build

# Or using python -m build (requires python3-venv on Debian/Ubuntu)
python -m build
```

This creates:
- `dist/palace_ai-X.Y.Z.tar.gz` (source distribution)
- `dist/palace_ai-X.Y.Z-py3-none-any.whl` (wheel)

## Testing the Build

Verify the package contents:

```bash
# Check the wheel contents
unzip -l dist/palace_ai-*.whl

# Verify package metadata
twine check dist/*
```

## Publishing to TestPyPI (Recommended First)

Always test on TestPyPI before publishing to production:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# You'll be prompted for:
# - username: __token__
# - password: your TestPyPI API token (starts with pypi-)
```

Test installation from TestPyPI:

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ palace-ai

# Test the installation
palace --help
pal --help
```

## Publishing to PyPI (Production)

Once tested, publish to production PyPI:

```bash
# Upload to PyPI
twine upload dist/*

# You'll be prompted for:
# - username: __token__
# - password: your PyPI API token (starts with pypi-)
```

## Using API Tokens in CI/CD

For automated publishing (GitHub Actions, etc.), use environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token-here
twine upload dist/*
```

## Version Management

Before publishing a new version:

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

2. Update `VERSION` constant in `palace.py`:
   ```python
   VERSION = "0.2.0"
   ```

3. Create a git tag:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

4. Rebuild and publish:
   ```bash
   rm -rf dist/
   uv build
   twine check dist/*
   twine upload dist/*
   ```

## Post-Publication

After publishing:

1. Verify on PyPI: https://pypi.org/project/palace-ai/
2. Test installation: `pip install palace-ai`
3. Update GitHub release notes
4. Announce on relevant channels

## Troubleshooting

### "File already exists" error
- You cannot overwrite existing versions on PyPI
- Increment the version number and rebuild

### Import errors after installation
- Check that `palace.py` has proper module structure
- Verify `__name__ == "__main__"` block for CLI entry point

### Missing files in package
- Check `MANIFEST.in` includes all necessary files
- Verify `pyproject.toml` has correct include/exclude patterns
- Use `twine check dist/*` to validate

### Authentication failures
- Ensure you're using `__token__` as username (not your PyPI username)
- Verify API token has upload permissions
- Check token hasn't expired

## Security Best Practices

1. **Never commit API tokens** to version control
2. **Use environment variables** for tokens in CI/CD
3. **Rotate tokens regularly** (at least annually)
4. **Use scoped tokens** (limit to specific projects when possible)
5. **Enable 2FA** on your PyPI account

## References

- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PEP 517 Build System](https://www.python.org/dev/peps/pep-0517/)
