# Contributing to Palace

Thank you for considering contributing to Palace! This document provides guidelines and best practices for contributing.

## Golden Rules

### 🥇 Rule #1: TDD (Test-Driven Development)

**EVERY feature, EVERY change, EVERY improvement MUST have tests.**

- Write tests FIRST, implementation follows
- Tests are modular and build up incrementally
- Use pytest for all testing
- Tests define the spec - they ARE the documentation
- No pull request without tests
- No commit to main without green tests

**Test coverage is not optional - it's mandatory.**

### 🥈 Rule #2: Don't Be Prescriptive

**OFFER OPTIONS, don't dictate solutions.**

- Present MULTIPLE valid paths forward, not just one "best" answer
- The user decides what to do - you suggest possibilities
- Never say "you should do X" - say "options include X, Y, Z"
- Don't restrict outputs artificially
- When suggesting actions, give MANY options across different categories
- Let the user steer - Palace is a tool, not a boss

**The user is in control. Always.**

## Development Setup

### Prerequisites

- Python 3.8+
- uv (recommended) or pip
- Git

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/your-org/palace.git
cd palace

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use uv (faster)
uv pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov

# Run tests to verify setup
pytest tests/
```

## Development Workflow

### 1. TDD Cycle

For every change:

```bash
# 1. Write a failing test first
vim tests/test_feature.py

# 2. Run the test (it should fail)
pytest tests/test_feature.py -v

# 3. Write minimal code to make it pass
vim palace.py

# 4. Run the test again (it should pass)
pytest tests/test_feature.py -v

# 5. Refactor if needed
# 6. Run ALL tests
pytest tests/ -v
```

### 2. Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=palace --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py -v

# Run specific test class
pytest tests/test_core.py::TestSessionManagement -v

# Run specific test
pytest tests/test_core.py::TestSessionManagement::test_save_and_load_session -v

# Run with verbose output
pytest tests/ -vv
```

### 3. Code Style

Palace follows these conventions:

- **PEP 8** for Python code style
- **Descriptive names** over brevity (e.g., `parse_action_selection` not `parse_sel`)
- **Docstrings** for all public methods
- **Type hints** where they add clarity
- **Comments** for complex logic only (code should be self-documenting)

### 4. Git Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and commit frequently
git add .
git commit -m "Add feature X with tests"

# Push to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

#### Commit Message Format

```
Brief description of the change

Detailed explanation of:
- What changed
- Why it changed
- Any breaking changes
- Test coverage

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Testing Guidelines

### Test Organization

```
tests/
├── __init__.py
├── test_core.py          # Core Palace functionality
├── test_mcp.py           # MCP server integration
├── test_modes.py         # Interactive vs non-interactive modes
├── test_prompts.py       # Prompt building and context
└── test_integration.py   # Integration tests (if needed)
```

### Writing Good Tests

```python
class TestFeature:
    """Test feature description"""

    @pytest.fixture
    def temp_palace(self, tmp_path):
        """Create a Palace instance in temp directory"""
        os.chdir(tmp_path)
        palace = Palace()
        palace.ensure_palace_dir()
        yield palace

    def test_specific_behavior(self, temp_palace):
        """Test does X when Y happens"""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = temp_palace.some_method(input_data)

        # Assert
        assert result is not None
        assert result["key"] == "expected"
```

### Test Naming

- Use descriptive names: `test_parse_space_separated_numbers`
- Not: `test_parser`, `test_1`, `test_case`
- Start with `test_`
- Describe the expected behavior

### Test Coverage Goals

- **Core functionality**: 100% coverage
- **Edge cases**: All edge cases tested
- **Error handling**: Error paths tested
- **Integration points**: MCP, CLI, file I/O tested

## Feature Development

### Adding a New Feature

1. **Write the test first**
   ```python
   def test_new_feature_does_x(temp_palace):
       """New feature should do X"""
       result = temp_palace.new_feature()
       assert result == expected
   ```

2. **Run the test (should fail)**
   ```bash
   pytest tests/test_core.py::test_new_feature_does_x
   ```

3. **Implement the feature**
   ```python
   def new_feature(self):
       """Implement new feature"""
       return expected_result
   ```

4. **Run the test (should pass)**

5. **Add more tests for edge cases**

6. **Document the feature**
   - Update CLAUDE.md if it affects Claude interaction
   - Update README.md if it's user-facing
   - Add docstrings to the code

### Modifying Existing Features

1. **Add test for new behavior** (should fail)
2. **Modify code**
3. **Ensure all existing tests still pass**
4. **Add tests for edge cases**
5. **Update documentation**

## MCP Server Development

Palace serves as both a CLI tool and an MCP server.

### Testing MCP Features

```python
def test_mcp_tool(temp_palace):
    """Test MCP tool behavior"""
    from palace import handle_permission

    result = handle_permission(
        tool_name="Bash",
        input={"command": "ls"}
    )

    assert result["approved"] is True
```

### MCP Server Installation

```bash
# Install Palace as MCP server globally
claude mcp add palace --scope user \
  /path/to/palace/.venv/bin/python \
  /path/to/palace/palace.py
```

## Documentation

### What to Document

- **Public APIs**: All public methods need docstrings
- **Features**: Update CLAUDE.md for Claude-facing features
- **User-facing changes**: Update README.md
- **Breaking changes**: Document in commit messages and CHANGELOG

### Documentation Style

```python
def method_name(self, param: str) -> dict:
    """
    Brief one-line description.

    Longer description if needed. Explain what the method does,
    when to use it, and any important details.

    Args:
        param: Description of parameter

    Returns:
        Description of return value

    Example:
        >>> palace = Palace()
        >>> result = palace.method_name("test")
        >>> print(result)
        {"key": "value"}
    """
```

## Pull Request Process

1. **Ensure all tests pass**
   ```bash
   pytest tests/ -v
   ```

2. **Check test coverage**
   ```bash
   pytest tests/ --cov=palace --cov-report=term-missing
   ```

3. **Verify code quality**
   - No unused imports
   - No dead code
   - Clear variable names
   - Proper error handling

4. **Update documentation**
   - Add/update docstrings
   - Update CLAUDE.md or README.md if needed
   - Add comments for complex logic

5. **Create pull request**
   - Clear title describing the change
   - Description includes:
     - What changed
     - Why it changed
     - How to test it
     - Any breaking changes
   - Link to related issues

6. **Respond to review feedback**
   - Address all comments
   - Push additional commits
   - Request re-review when ready

## Common Tasks

### Adding a New Command

1. **Write tests**
   ```python
   def test_new_command(temp_palace):
       """Test new command behavior"""
       # Test the command
   ```

2. **Add command method**
   ```python
   def cmd_new_command(self, args):
       """Execute new command"""
       pass
   ```

3. **Register in main()**
   ```python
   commands = {
       'new-command': palace.cmd_new_command,
       # ...
   }
   ```

4. **Add argparse configuration**
   ```python
   parser_new = subparsers.add_parser('new-command', help='Description')
   ```

5. **Create slash command** (if appropriate)
   ```bash
   # In .claude/commands/pal-new-command.md
   ```

### Adding a New Test Suite

1. **Create test file**
   ```bash
   touch tests/test_new_feature.py
   ```

2. **Add test class**
   ```python
   class TestNewFeature:
       """Test new feature"""

       @pytest.fixture
       def setup(self):
           """Setup for tests"""
           pass
   ```

3. **Write tests**

4. **Run tests**
   ```bash
   pytest tests/test_new_feature.py -v
   ```

## Questions?

- **File an issue**: For bugs or feature requests
- **Start a discussion**: For questions or ideas
- **Check the docs**: CLAUDE.md explains the architecture

## License

By contributing, you agree that your contributions will be licensed under the same license as Palace.
