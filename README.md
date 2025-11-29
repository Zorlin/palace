# palace
structure determines action

Palace is an experimental, self-hosted agent wrapper for Claude Code and the Claude Agent SDK.
It aims to give Claude access to embedded knowledge from many domains of expertise,
utilising all of the tools, tricks and tools available to Claude 

It can operate as a standalone agent (using the SDK) or as a wrapper around Claude Code.

It can also install its plugins into Claude Code for direct operation.

## Installation
You'll need just a few things to get started with Palace.

1. A copy of Python
2. Node.js
3. Claude Code

## Usage
Once you've installed Palace, you can use it by running the following command:

```
python3 palace.py <command>
```

The following main commands are available:

Creating a project:
- `pal next` - Suggest what to do next in the project
- `pal new` - Create a project
- `pal scaffold` - Scaffold a project with best practices
- `pal test` - Test the project

Installation/uninstallation:
- `pal install` - Install all of the Palace plugins in Claude Code

## Contributing
Please follow Semantic Line Breaks (SemBr).

We welcome contributions to Palace! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes - ensure all existing tests pass,
   and ideally add new tests for your changes
4. Submit a pull request

## License
Palace is licensed under the GNU Affero General Public License v3.0. See the LICENSE file for more information.
