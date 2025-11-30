# Error Recovery in Palace RHSI Loop

## Current State

Palace currently has basic error handling:
- `exit_code != 0` logs warning and continues
- Failed Claude invocations return error code
- No automatic retry logic
- No graceful degradation
- Sessions saved at start, not after each successful iteration

## Error Scenarios

### 1. Claude Code Failures

**Symptoms:**
- Non-zero exit code from `claude` command
- Claude process crashes
- API rate limits
- Network timeouts

**Current Handling:**
```python
if exit_code != 0:
    print(f"\n⚠️  Claude exited with code {exit_code}")
# Continues to next iteration
```

**Issues:**
- No distinction between recoverable and fatal errors
- No retry logic
- May continue with corrupted state

### 2. Permission Denials

**Symptoms:**
- MCP permission handler denies operation
- User interrupts during dangerous operation
- Safety assessment blocks command

**Current Handling:**
- Logged to history
- No automatic recovery
- Loop may continue with incomplete task

### 3. Tool Execution Failures

**Symptoms:**
- File operation fails (permissions, disk full)
- Git command fails (conflicts, authentication)
- Test failures
- Build errors

**Current Handling:**
- Relies on Claude's error handling
- No systematic recovery
- May accumulate failed state

### 4. Session Corruption

**Symptoms:**
- Session file missing or malformed
- Invalid action selection
- Corrupted history log

**Current Handling:**
- Session load returns None
- Basic validation on load
- No repair mechanisms

## Proposed Error Recovery Strategies

### Strategy 1: Retry with Backoff

For transient failures (network, rate limits):

```python
def invoke_claude_with_retry(self, prompt: str, max_retries: int = 3):
    """Invoke Claude with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            exit_code, actions = self.invoke_claude(prompt)
            if exit_code == 0:
                return exit_code, actions

            # Exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"⚠️  Error: {e}, retrying...")

    return 1, None  # All retries failed
```

### Strategy 2: Session Checkpointing

Save session state after each successful iteration:

```python
def cmd_next(self, args):
    """RHSI loop with checkpointing"""
    while True:
        iteration += 1

        # Checkpoint before iteration
        self.save_session(session_id, {
            "iteration": iteration,
            "pending_actions": pending_actions,
            "last_prompt": current_prompt,
            "context_snapshot": context
        })

        # Execute iteration
        try:
            exit_code, selected_actions = self.invoke_claude(current_prompt)

            # Checkpoint after success
            self.save_session(session_id, {
                "iteration": iteration,
                "completed": True,
                "selected_actions": selected_actions
            })

        except Exception as e:
            # Restore from checkpoint
            print(f"❌ Error: {e}")
            print(f"💾 Session saved at iteration {iteration}")
            print(f"Resume with: palace next --resume {session_id}")
            break
```

### Strategy 3: Graceful Degradation

Fall back to simpler modes on repeated failures:

```python
def handle_claude_failure(self, attempt: int, error: Exception):
    """Handle failures with graceful degradation"""
    if attempt == 0:
        # First failure: retry with same config
        return "retry"
    elif attempt == 1:
        # Second failure: try without streaming
        print("⚠️  Disabling streaming mode...")
        return "no-stream"
    elif attempt == 2:
        # Third failure: fallback to prompt file only
        print("⚠️  Falling back to non-interactive mode...")
        return "prompt-file"
    else:
        # Fatal: save state and exit
        print("❌ Fatal error, saving session...")
        return "fatal"
```

### Strategy 4: Error Context Preservation

Include error information in next iteration:

```python
def _build_error_recovery_prompt(self, error_info: dict) -> str:
    """Build prompt with error context"""
    return f"""Previous iteration failed with error:

Error: {error_info['message']}
Exit Code: {error_info['exit_code']}
Last Action: {error_info['last_action']}

Please analyze the error and suggest:
1. What went wrong
2. How to recover
3. Alternative approaches

Then continue with the task or suggest next steps."""
```

### Strategy 5: Safety Rollback

Rollback dangerous operations that fail:

```python
def execute_with_rollback(self, command: str, backup_tag: str = None):
    """Execute command with automatic rollback on failure"""
    # Create backup
    if backup_tag:
        subprocess.run(["git", "tag", backup_tag])

    try:
        result = subprocess.run(command, shell=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        # Rollback on failure
        if backup_tag:
            print(f"🔄 Rolling back to {backup_tag}")
            subprocess.run(["git", "reset", "--hard", backup_tag])
        raise
```

## Implementation Plan

### Phase 1: Basic Retry Logic

- Add retry with exponential backoff
- Distinguish transient vs permanent errors
- Log retry attempts to history

### Phase 2: Session Checkpointing

- Save session after each successful iteration
- Include enough context to resume
- Validate session integrity on load

### Phase 3: Error Context

- Include error information in next prompt
- Allow Claude to analyze and recover
- Track recovery attempts per session

### Phase 4: Graceful Degradation

- Detect repeated failures
- Fall back to simpler modes
- Preserve partial progress

### Phase 5: Safety Mechanisms

- Rollback on critical failures
- Validate state before continuing
- Alert user on dangerous operations

## Error Recovery Heuristics

### When to Retry

✅ **Retry:**
- Network timeouts
- Rate limit errors (429)
- Temporary file locks
- Transient API errors

❌ **Don't Retry:**
- Permission denied (403)
- Invalid syntax errors
- Malformed requests
- User interrupts (Ctrl-C)

### When to Rollback

✅ **Rollback:**
- Failed git history rewrite
- Database migration errors
- Destructive file operations that failed mid-way

❌ **Don't Rollback:**
- Test failures (keep changes for debugging)
- Linting errors (keep code to fix)
- Minor file operation errors

### When to Exit

✅ **Exit Gracefully:**
- User cancels (Ctrl-C)
- Fatal configuration error
- Repeated failures (3+ attempts)
- Critical safety violation

❌ **Don't Exit:**
- Single test failure
- Minor tool errors
- Non-critical warnings

## Monitoring and Telemetry

Track error patterns to improve recovery:

```python
error_stats = {
    "total_errors": 0,
    "errors_by_type": Counter(),
    "recovery_attempts": 0,
    "successful_recoveries": 0,
    "fatal_errors": 0
}

def log_error(self, error_type: str, recovered: bool):
    """Log error for analysis"""
    self.log_action("error", {
        "type": error_type,
        "recovered": recovered,
        "timestamp": time.time()
    })
```

## Testing Error Recovery

```python
class TestErrorRecovery:
    """Test error recovery mechanisms"""

    def test_retry_on_network_error(self):
        """Verify retry logic for network errors"""
        # Mock network failure, then success
        pass

    def test_session_restore_after_crash(self):
        """Verify session can be restored after failure"""
        pass

    def test_graceful_degradation(self):
        """Verify fallback to simpler modes"""
        pass

    def test_error_context_in_prompt(self):
        """Verify error info included in recovery prompt"""
        pass
```

## User Experience

### Good Error Messages

```
❌ Claude invocation failed (exit code: 1)
💾 Session saved: pal-abc123
🔄 Retry attempt 1/3 in 2 seconds...
```

```
❌ Network timeout after 30s
💾 Progress saved to: .palace/sessions/pal-abc123.json
▶️  Resume with: palace next --resume pal-abc123
```

### Recovery Guidance

When errors occur, provide clear next steps:

```
❌ Error during RHSI iteration 5

What happened:
  Claude Code returned exit code 1
  Last action: "Run integration tests"

Your options:
  1. Retry: palace next --resume pal-abc123
  2. Skip this action: palace next --resume pal-abc123 --skip 1
  3. Review logs: cat .palace/history.jsonl
  4. Start fresh: palace next
```

## Security Considerations

### Safe Recovery

- Never retry dangerous operations without confirmation
- Validate state before resuming
- Don't auto-retry credential operations
- Alert user to security-sensitive failures

### Rollback Safety

- Only rollback if backup confirmed
- Don't rollback across user decisions
- Preserve audit trail of rollbacks
- Require explicit approval for destructive rollbacks

## Future Enhancements

1. **Intelligent Recovery:** Use Haiku to analyze errors and suggest fixes
2. **Pattern Learning:** Learn from errors to prevent recurrence
3. **Distributed Resilience:** Handle multi-machine RHSI loops
4. **Partial Progress:** Save work-in-progress for partial recovery

## Conclusion

Error recovery transforms Palace from a fragile orchestrator into a resilient RHSI platform. By combining retry logic, checkpointing, graceful degradation, and intelligent recovery, Palace can continue making progress even when individual operations fail.

**Priority Implementation:**
1. Session checkpointing (critical for long RHSI loops)
2. Basic retry with backoff (handles most transient errors)
3. Error context in prompts (enables Claude-driven recovery)
