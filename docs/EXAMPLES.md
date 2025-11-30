# Palace Examples

Real-world examples of using Palace for different workflows.

## Table of Contents

1. [Starting a New Project](#starting-a-new-project)
2. [Adding a Feature](#adding-a-feature)
3. [Refactoring Legacy Code](#refactoring-legacy-code)
4. [Debugging and Fixing](#debugging-and-fixing)
5. [Documentation Workflow](#documentation-workflow)
6. [Release Preparation](#release-preparation)
7. [Team Collaboration](#team-collaboration)

---

## Starting a New Project

### Scenario: Create a new Python CLI tool

```bash
$ python palace.py new my-cli-tool
```

**Palace prompts Claude:**

```
Create a new project called 'my-cli-tool'.

Steps:
1. Create the project directory
2. Initialize Palace (.palace/config.json)
3. Create initial files (README.md, etc.)
4. Set up version control if appropriate
5. Create a SPEC.md outlining what this project should do

Ask the user questions if you need more information.
```

**Claude asks:**

```
What should my-cli-tool do? Here are some questions:

1. What is the primary purpose?
2. What language/framework? (Python suggested)
3. Will it need configuration files?
4. Should it have subcommands or flags?
```

**You respond:**

```
1. A CLI for managing TODO lists
2. Python with Click
3. Yes, YAML config
4. Subcommands: add, list, done, remove
```

**Claude creates:**

```
✅ Created directory: my-cli-tool/
✅ Created: README.md, SPEC.md, pyproject.toml
✅ Created: src/my_cli_tool/__init__.py, main.py
✅ Created: tests/test_main.py
✅ Initialized git repository
✅ Palace initialized

Next: Run tests and start implementing!
```

**Continue with RHSI:**

```bash
$ cd my-cli-tool
$ claude

> /pal-next

ACTIONS:
1. Implement the 'add' command
2. Write tests for 'add' command
3. Set up CI/CD pipeline
4. Add example to README

Select: 2  # TDD first!

[Claude writes comprehensive tests]

> /pal-next

ACTIONS:
1. Implement the 'add' command (tests already written)
2. Add example to README
3. Set up CI/CD pipeline

Select: 1

[Claude implements to pass the tests]
```

---

## Adding a Feature

### Scenario: Add user authentication to a web app

**Current state:**
- Express.js API
- No authentication
- Several unprotected endpoints

```bash
$ cd my-api
$ claude

> /pal-next

[Claude analyzes the project]

ACTIONS:
1. Add user authentication with JWT
2. Fix failing tests in user.test.js
3. Update API documentation
4. Optimize database queries

Select: 1
```

**With modifiers:**

```
> /pal-next

ACTIONS:
1. Add user authentication with JWT
2. Fix failing tests
3. Update API docs

Select: 1 (use passport.js and bcrypt)

[Claude implements auth using specified libraries]
```

**Next iteration - Tests:**

```
> /pal-next

[Claude knows auth was just added]

ACTIONS:
1. Write comprehensive tests for auth flow
   - Registration
   - Login
   - Token validation
   - Password hashing

2. Update existing endpoint tests to include auth headers
3. Fix failing tests (from before)
4. Update API documentation with auth examples

Select: 1 2

[Claude writes tests AND updates existing tests]
```

**Next iteration - Documentation:**

```
> /pal-next

ACTIONS:
1. Document authentication in README
   - How to register
   - How to login
   - How to use tokens

2. Add Postman collection with auth examples
3. Update OpenAPI spec

Select: 1-3

[Claude completes all documentation tasks]
```

**Final iteration - Security review:**

```
> /pal-next

ACTIONS:
1. Security audit of auth implementation
   - Check for common vulnerabilities
   - Validate token expiry
   - Test password requirements

2. Add rate limiting to auth endpoints
3. Set up session management

Select: 1

[Claude performs security review, suggests improvements]
```

---

## Refactoring Legacy Code

### Scenario: Modernize old JavaScript codebase

**Current state:**
- jQuery spaghetti code
- No module system
- No tests
- Mixed var/let/const

**Session export** to track the multi-day refactoring:

```bash
$ python palace.py init
$ claude

> /pal-next

[Claude analyzes the mess]

ACTIONS:
1. Audit current code structure
   - Identify global variables
   - Find duplicated code
   - Map dependencies

2. Create test harness for existing functionality
3. Set up modern build pipeline (Webpack/Vite)
4. Migrate to ES6 modules

Select: 1

[Claude creates detailed audit report]

# Export session for tomorrow
$ python palace.py export pal-abc123 -o refactor-day1.json
```

**Day 2 - Resume session:**

```bash
$ python palace.py next --resume pal-abc123

[Picks up from audit]

ACTIONS:
1. Create test harness for existing functionality
   - Identify critical paths
   - Write characterization tests

2. Set up modern build pipeline
3. Migrate to ES6 modules

Select: 1

[Claude writes tests that lock in current behavior]
```

**Day 3 - Actual refactoring:**

```bash
> /pal-next

ACTIONS:
1. Migrate jQuery to vanilla JS
   - One module at a time
   - Keep tests green

2. Extract reusable components
3. Remove dead code
4. Update documentation

Select: 1 (start with utils.js)

[Claude refactors utils.js, tests pass]

> /pal-next

ACTIONS:
1. Continue jQuery migration (next: forms.js)
2. Extract reusable components
3. Remove dead code

Select: 1

[Claude continues methodically]
```

---

## Debugging and Fixing

### Scenario: Production bug - users can't log in

**Quick fix workflow:**

```bash
$ git checkout -b hotfix-login
$ claude

> Investigate why users can't log in. Check recent changes to auth module.

[Claude analyzes, finds issue]

> /pal-next

ACTIONS:
1. Fix the authentication bug
   - Issue: JWT secret was changed in env
   - Solution: Add fallback key lookup

2. Add test to prevent regression
3. Deploy hotfix to staging
4. Notify team

Select: 1 2

[Claude fixes bug and adds regression test]

> /pal-next

ACTIONS:
1. Deploy hotfix to staging
2. Verify fix in staging
3. Create pull request
4. Update incident log

Select: 1-4

[Claude executes deploy workflow]
```

**Root cause analysis:**

```
> /pal-next

ACTIONS:
1. Root cause analysis
   - Why wasn't this caught in tests?
   - What process change prevents this?

2. Update deployment checklist
3. Add monitoring for JWT errors

Select: 1

[Claude writes detailed RCA]
```

---

## Documentation Workflow

### Scenario: Document an undocumented API

```bash
> /pal-next

ACTIONS:
1. Generate API documentation from code
   - Extract JSDoc comments
   - Create OpenAPI spec
   - Generate examples

2. Write usage guide
3. Create tutorial
4. Add to docs site

Select: 1

[Claude scans code, generates docs]

> /pal-next

ACTIONS:
1. Review and improve generated docs
   - Add missing descriptions
   - Improve examples
   - Fix formatting

2. Write usage guide for common workflows
3. Create getting started tutorial

Select: 2

[Claude writes comprehensive usage guide]

> /pal-next

ACTIONS:
1. Create getting started tutorial
   - Step by step walkthrough
   - Real-world example
   - Troubleshooting section

2. Add docs to website
3. Update README with links

Select: 1-3

[Claude creates tutorial and publishes docs]
```

---

## Release Preparation

### Scenario: Prepare v2.0.0 release

**Week before release:**

```bash
$ git checkout develop
$ claude

> /pal-next

ACTIONS:
1. Audit CHANGELOG
   - Verify all features listed
   - Add breaking changes section
   - Update migration guide

2. Run full test suite
3. Update version numbers
4. Review documentation

Select: 1

[Claude reviews and updates CHANGELOG]

> /pal-next

ACTIONS:
1. Run full test suite
   - Unit tests
   - Integration tests
   - E2E tests

2. Check for deprecation warnings
3. Verify examples still work

Select: 1-3

[Claude runs comprehensive tests]
```

**Day before release:**

```
> /pal-next

ACTIONS:
1. Create release branch
2. Update version to 2.0.0
3. Build release artifacts
4. Test installation process

Select: 1-4

[Claude prepares release]

> /pal-next

ACTIONS:
1. Draft release notes
   - Highlight major features
   - List breaking changes
   - Include upgrade instructions

2. Tag release
3. Push to repository
4. Trigger CI/CD

Select: 1

[Claude drafts comprehensive release notes]
```

**Release day:**

```
> /pal-next

ACTIONS:
1. Review final release notes
2. Publish release
3. Update documentation site
4. Announce on social media
5. Monitor for issues

Select: 1-5

[Claude executes release workflow]
```

---

## Team Collaboration

### Scenario: Onboarding new team member

**Export your workflow:**

```bash
# You've built a feature following best practices
$ claude

> /pal-next
Select: [various tasks]

> /pal-next
Select: [more tasks]

# Export the session
$ python palace.py export pal-abc123 -o feature-workflow.json
```

**New team member imports:**

```bash
$ python palace.py import feature-workflow.json

# They can see your exact workflow
$ python palace.py next --resume pal-xyz789

[Shows the sequence of actions you took]
```

### Scenario: Code review assistance

**Reviewer uses Palace:**

```bash
$ git checkout pr-branch
$ claude

> /pal-next

[Claude analyzes the PR]

ACTIONS:
1. Review code changes
   - Check for bugs
   - Verify tests
   - Assess style

2. Run tests locally
3. Suggest improvements
4. Approve or request changes

Select: 1

[Claude provides detailed review]

ACTIONS:
1. Test the new feature manually
2. Check edge cases
3. Verify documentation updated
4. Leave review comments

Select: 1-4
```

### Scenario: Pair programming with Palace

**Two developers, one session:**

```bash
# Developer A
$ claude

> /pal-next

ACTIONS:
1. Implement user registration
2. Add email validation
3. Set up database schema

Select: 1

[Developer A implements]

# Export session
$ python palace.py export pal-abc123 -o pair-session.json

# Share with Developer B
$ scp pair-session.json dev-b@host:/tmp/
```

**Developer B continues:**

```bash
$ python palace.py import /tmp/pair-session.json

$ python palace.py next --resume pal-xyz789

ACTIONS:
1. Add email validation (Developer A's pending task)
2. Set up database schema (Developer A's pending task)
3. Write tests for registration

Select: 1-3

[Developer B picks up where A left off]
```

---

## Advanced Patterns

### Using Masks for Specialized Workflows

**Security audit with security-expert mask:**

```bash
> /pal-next --mask security-expert

ACTIONS:
1. Security audit of authentication
2. Check for SQL injection vulnerabilities
3. Review CORS configuration
4. Validate input sanitization
5. Test for XSS vulnerabilities

Select: 1-5

[Claude performs thorough security review]
```

**Performance optimization with performance-expert mask:**

```bash
> /pal-next --mask performance-expert

ACTIONS:
1. Profile application performance
2. Identify bottlenecks
3. Optimize database queries
4. Add caching layer
5. Benchmark improvements

Select: 1

[Claude analyzes performance]
```

### Composing Multiple Masks

**Comprehensive feature implementation:**

```bash
> /pal-next --masks "tdd-expert,security-expert" --strategy layer

ACTIONS:
1. Write tests first (TDD approach)
2. Implement with security in mind
3. Security audit implementation
4. Performance benchmarks

Select: 1-4

[Claude follows both TDD and security best practices]
```

---

## Summary

Palace adapts to YOUR workflow:

- **New projects**: Scaffolding and setup
- **Features**: TDD-driven implementation
- **Refactoring**: Methodical, test-safe migrations
- **Debugging**: Fast iteration to find and fix
- **Docs**: Comprehensive documentation generation
- **Releases**: Systematic preparation
- **Collaboration**: Shareable workflows

The common pattern: `/pal-next`, select, execute, repeat.

Each iteration learns from the last, making suggestions smarter over time.

🏛️ **Structure determines action.**
