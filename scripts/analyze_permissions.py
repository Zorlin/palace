#!/usr/bin/env python3
"""
Analyze Palace permission logs

Reads history.jsonl and analyzes permission requests and decisions.

Usage:
    python scripts/analyze_permissions.py
    python scripts/analyze_permissions.py --detailed
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from palace import Palace


def load_permission_history(palace: Palace):
    """Load permission-related entries from history"""
    history_file = palace.palace_dir / "history.jsonl"

    if not history_file.exists():
        return []

    permissions = []
    with open(history_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("action") in ("permission_request", "permission_decision"):
                    permissions.append(entry)
            except json.JSONDecodeError:
                continue

    return permissions


def analyze_permission_patterns(permissions):
    """Analyze patterns in permission requests"""
    print("📊 Permission Request Patterns")
    print("=" * 60)

    # Count requests by tool
    tool_counts = Counter()
    approval_counts = defaultdict(lambda: {"approved": 0, "denied": 0})

    requests = [p for p in permissions if p["action"] == "permission_request"]
    decisions = [p for p in permissions if p["action"] == "permission_decision"]

    for req in requests:
        details = req.get("details", {}).get("request", {})
        tool_name = details.get("tool_name", "unknown")
        tool_counts[tool_name] += 1

    print(f"Total requests: {len(requests)}")
    print(f"Total decisions: {len(decisions)}")
    print()

    if tool_counts:
        print("Requests by tool:")
        for tool, count in tool_counts.most_common():
            print(f"  {tool}: {count}")
        print()

    # Analyze decisions
    for dec in decisions:
        details = dec.get("details", {})
        tool_name = details.get("tool_name", "unknown")
        approved = details.get("approved", False)

        if approved:
            approval_counts[tool_name]["approved"] += 1
        else:
            approval_counts[tool_name]["denied"] += 1

    if approval_counts:
        print("Approval rates by tool:")
        for tool in sorted(approval_counts.keys()):
            stats = approval_counts[tool]
            total = stats["approved"] + stats["denied"]
            rate = stats["approved"] / total * 100 if total > 0 else 0
            print(f"  {tool}: {stats['approved']}/{total} ({rate:.1f}% approved)")
        print()


def analyze_command_patterns(permissions):
    """Analyze command patterns in Bash tool requests"""
    print("💻 Bash Command Patterns")
    print("=" * 60)

    bash_requests = []
    for p in permissions:
        if p.get("action") == "permission_request":
            details = p.get("details", {}).get("request", {})
            if details.get("tool_name") == "Bash":
                cmd = details.get("input", {}).get("command", "")
                if cmd:
                    bash_requests.append(cmd)

    if not bash_requests:
        print("No Bash commands found in history")
        print()
        return

    print(f"Total Bash commands: {len(bash_requests)}")
    print()

    # Categorize commands
    categories = {
        "git": [],
        "python": [],
        "file_ops": [],
        "package_mgmt": [],
        "other": []
    }

    for cmd in bash_requests:
        if cmd.startswith("git "):
            categories["git"].append(cmd)
        elif "python" in cmd or "pytest" in cmd:
            categories["python"].append(cmd)
        elif any(x in cmd for x in ["rm ", "mv ", "cp ", "mkdir ", "touch "]):
            categories["file_ops"].append(cmd)
        elif any(x in cmd for x in ["pip ", "npm ", "uv ", "apt ", "brew "]):
            categories["package_mgmt"].append(cmd)
        else:
            categories["other"].append(cmd)

    print("Commands by category:")
    for cat, cmds in categories.items():
        if cmds:
            print(f"  {cat}: {len(cmds)}")
    print()


def analyze_safety_decisions(permissions):
    """Analyze safety assessment decisions"""
    print("🛡️  Safety Decisions")
    print("=" * 60)

    decisions = [p for p in permissions if p.get("action") == "permission_decision"]

    if not decisions:
        print("No safety decisions found in history")
        print()
        return

    print(f"Total decisions: {len(decisions)}")
    print()

    # Count approvals/denials
    approved = sum(1 for d in decisions if d.get("details", {}).get("approved", False))
    denied = len(decisions) - approved

    print(f"Approved: {approved} ({approved/len(decisions)*100:.1f}%)")
    print(f"Denied: {denied} ({denied/len(decisions)*100:.1f}%)")
    print()

    # Show denial reasons
    denials = [d for d in decisions if not d.get("details", {}).get("approved", False)]
    if denials:
        print("Denial reasons:")
        for d in denials[:5]:  # Show first 5
            reason = d.get("details", {}).get("reason", "No reason provided")
            tool = d.get("details", {}).get("tool_name", "unknown")
            print(f"  [{tool}] {reason}")
        if len(denials) > 5:
            print(f"  ... and {len(denials) - 5} more")
        print()


def show_timeline(permissions, limit=10):
    """Show timeline of recent permission activity"""
    print("🕒 Recent Permission Activity")
    print("=" * 60)

    if not permissions:
        print("No permission activity found")
        print()
        return

    recent = sorted(permissions, key=lambda p: p.get("timestamp", 0), reverse=True)[:limit]

    for p in reversed(recent):
        ts = p.get("timestamp", 0)
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        action = p.get("action", "unknown")

        if action == "permission_request":
            details = p.get("details", {}).get("request", {})
            tool = details.get("tool_name", "unknown")
            print(f"  {dt} | REQUEST  | {tool}")
        elif action == "permission_decision":
            details = p.get("details", {})
            tool = details.get("tool_name", "unknown")
            approved = details.get("approved", False)
            status = "APPROVED" if approved else "DENIED"
            print(f"  {dt} | {status:8} | {tool}")

    print()


def detailed_report(permissions):
    """Show detailed information about each permission"""
    print("📋 Detailed Permission Log")
    print("=" * 60)

    if not permissions:
        print("No permissions to report")
        print()
        return

    for i, p in enumerate(permissions, 1):
        ts = p.get("timestamp", 0)
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        action = p.get("action", "unknown")

        print(f"{i}. {dt} - {action}")

        if action == "permission_request":
            details = p.get("details", {}).get("request", {})
            print(f"   Tool: {details.get('tool_name', 'unknown')}")
            print(f"   Input: {json.dumps(details.get('input', {}), indent=6)}")
        elif action == "permission_decision":
            details = p.get("details", {})
            print(f"   Tool: {details.get('tool_name', 'unknown')}")
            print(f"   Approved: {details.get('approved', False)}")
            if details.get("reason"):
                print(f"   Reason: {details.get('reason')}")

        print()


def main():
    """Main analysis"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Palace permission logs")
    parser.add_argument("--detailed", action="store_true", help="Show detailed log")
    parser.add_argument("--limit", type=int, default=10, help="Timeline limit")
    args = parser.parse_args()

    print()
    print("🏛️  Palace Permission Log Analysis")
    print("=" * 60)
    print()

    palace = Palace()
    permissions = load_permission_history(palace)

    if not permissions:
        print("No permission logs found in history.")
        print()
        print("Permission logging happens when:")
        print("  - Claude Code calls Palace's MCP handle_permission tool")
        print("  - Palace makes safety decisions using Haiku")
        print()
        return

    # Run analyses
    analyze_permission_patterns(permissions)
    analyze_command_patterns(permissions)
    analyze_safety_decisions(permissions)
    show_timeline(permissions, limit=args.limit)

    if args.detailed:
        detailed_report(permissions)

    print("✅ Analysis complete!")
    print()


if __name__ == "__main__":
    main()
