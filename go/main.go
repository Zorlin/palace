package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Request from Palace
type Request struct {
	Prompt      string `json:"prompt"`
	System      string `json:"system,omitempty"`
	Model       string `json:"model,omitempty"`
	BaseURL     string `json:"base_url,omitempty"`
	WorkDir     string `json:"work_dir,omitempty"`
	EnableTools bool   `json:"enable_tools,omitempty"`
}

// Event types sent back to Palace
type Event struct {
	Type   string `json:"type"`
	Text   string `json:"text,omitempty"`
	Error  string `json:"error,omitempty"`
	Tool   string `json:"tool,omitempty"`
	Detail string `json:"detail,omitempty"`
}

var workDir string

func main() {
	// Read request from stdin
	reader := bufio.NewReader(os.Stdin)
	line, err := reader.ReadString('\n')
	if err != nil {
		sendEvent(Event{Type: "error", Error: fmt.Sprintf("failed to read input: %v", err)})
		os.Exit(1)
	}

	var req Request
	if err := json.Unmarshal([]byte(line), &req); err != nil {
		sendEvent(Event{Type: "error", Error: fmt.Sprintf("invalid JSON: %v", err)})
		os.Exit(1)
	}

	// Set working directory
	workDir = req.WorkDir
	if workDir == "" {
		workDir, _ = os.Getwd()
	}

	// Get API key
	apiKey := os.Getenv("ZAI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		sendEvent(Event{Type: "error", Error: "ZAI_API_KEY or ANTHROPIC_API_KEY environment variable not set"})
		os.Exit(1)
	}

	// Build client options
	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	baseURL := req.BaseURL
	if baseURL == "" {
		baseURL = "https://api.z.ai/api/anthropic"
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	client := anthropic.NewClient(opts...)

	// Determine model
	model := anthropic.ModelClaudeSonnet4_5_20250929
	if req.Model != "" {
		model = anthropic.Model(req.Model)
	}

	// Build initial messages
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(req.Prompt)),
	}

	// Define tools if enabled
	var tools []anthropic.ToolUnionParam
	if req.EnableTools {
		tools = getTools()
	}

	ctx := context.Background()

	// Agentic loop
	for {
		params := anthropic.MessageNewParams{
			Model:     model,
			MaxTokens: 8192,
			Messages:  messages,
		}

		if req.System != "" {
			params.System = []anthropic.TextBlockParam{
				{Text: req.System},
			}
		}

		if len(tools) > 0 {
			params.Tools = tools
		}

		// Make the API call (non-streaming for tool use)
		resp, err := client.Messages.New(ctx, params)
		if err != nil {
			sendEvent(Event{Type: "error", Error: err.Error()})
			os.Exit(1)
		}

		// Process response content
		var toolResults []anthropic.ContentBlockParamUnion
		hasToolUse := false

		for _, block := range resp.Content {
			switch block.Type {
			case "text":
				sendEvent(Event{Type: "text", Text: block.Text})

			case "tool_use":
				hasToolUse = true
				sendEvent(Event{Type: "tool_start", Tool: block.Name, Detail: string(block.ID)})

				// Execute the tool
				result := executeTool(block.Name, block.Input)
				sendEvent(Event{Type: "tool_result", Tool: block.Name, Detail: truncate(result, 200)})

				toolResults = append(toolResults, anthropic.NewToolResultBlock(
					block.ID,
					result,
					false,
				))
			}
		}

		// If there were tool uses, add assistant message and tool results, continue loop
		if hasToolUse {
			// Convert response content to param format
			var assistantContent []anthropic.ContentBlockParamUnion
			for _, block := range resp.Content {
				switch block.Type {
				case "text":
					assistantContent = append(assistantContent, anthropic.NewTextBlock(block.Text))
				case "tool_use":
					assistantContent = append(assistantContent, anthropic.ContentBlockParamUnion{
						OfToolUse: &anthropic.ToolUseBlockParam{
							ID:    block.ID,
							Name:  block.Name,
							Input: block.Input,
						},
					})
				}
			}

			// Add assistant's response to messages
			messages = append(messages, anthropic.MessageParam{
				Role:    "assistant",
				Content: assistantContent,
			})

			// Add tool results
			messages = append(messages, anthropic.MessageParam{
				Role:    "user",
				Content: toolResults,
			})
			continue
		}

		// No tool use and stop_reason is end_turn - we're done
		if resp.StopReason == "end_turn" {
			break
		}

		// Safety break
		break
	}

	sendEvent(Event{Type: "done"})
}

func getTools() []anthropic.ToolUnionParam {
	return []anthropic.ToolUnionParam{
		anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        "read_file",
				Description: anthropic.String("Read the contents of a file"),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "Path to the file (relative to project root)",
						},
					},
					Required: []string{"path"},
				},
			},
		},
		anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        "list_files",
				Description: anthropic.String("List files in a directory"),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "Directory path (relative to project root)",
						},
						"pattern": map[string]interface{}{
							"type":        "string",
							"description": "Optional glob pattern to filter files",
						},
					},
					Required: []string{"path"},
				},
			},
		},
		anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        "grep",
				Description: anthropic.String("Search for a pattern in files"),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"pattern": map[string]interface{}{
							"type":        "string",
							"description": "Search pattern (regex)",
						},
						"path": map[string]interface{}{
							"type":        "string",
							"description": "Directory or file to search in",
						},
					},
					Required: []string{"pattern"},
				},
			},
		},
		anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        "file_tree",
				Description: anthropic.String("Get a tree view of the project structure"),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: map[string]interface{}{
						"max_depth": map[string]interface{}{
							"type":        "integer",
							"description": "Maximum depth to traverse (default 3)",
						},
					},
					Required: []string{},
				},
			},
		},
	}
}

func executeTool(name string, input json.RawMessage) string {
	var params map[string]interface{}
	json.Unmarshal(input, &params)

	switch name {
	case "read_file":
		path := getString(params, "path")
		return readFile(path)

	case "list_files":
		path := getString(params, "path")
		pattern := getString(params, "pattern")
		return listFiles(path, pattern)

	case "grep":
		pattern := getString(params, "pattern")
		path := getString(params, "path")
		return grepFiles(pattern, path)

	case "file_tree":
		depth := getInt(params, "max_depth", 3)
		return fileTree(depth)

	default:
		return fmt.Sprintf("Unknown tool: %s", name)
	}
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getInt(m map[string]interface{}, key string, def int) int {
	if v, ok := m[key]; ok {
		if f, ok := v.(float64); ok {
			return int(f)
		}
	}
	return def
}

func readFile(path string) string {
	fullPath := filepath.Join(workDir, path)
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return fmt.Sprintf("Error reading file: %v", err)
	}
	content := string(data)
	// Limit size
	if len(content) > 50000 {
		content = content[:50000] + "\n... (truncated)"
	}
	return content
}

func listFiles(path, pattern string) string {
	dir := filepath.Join(workDir, path)
	var files []string

	err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		// Skip hidden dirs and common noise
		if info.IsDir() {
			name := info.Name()
			if strings.HasPrefix(name, ".") || name == "node_modules" || name == "target" || name == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}

		rel, _ := filepath.Rel(workDir, p)
		if pattern != "" {
			if matched, _ := filepath.Match(pattern, info.Name()); !matched {
				return nil
			}
		}
		files = append(files, rel)
		if len(files) > 200 {
			return fmt.Errorf("limit reached")
		}
		return nil
	})

	if err != nil && len(files) == 0 {
		return fmt.Sprintf("Error: %v", err)
	}

	return strings.Join(files, "\n")
}

func grepFiles(pattern, path string) string {
	searchPath := workDir
	if path != "" {
		searchPath = filepath.Join(workDir, path)
	}

	cmd := exec.Command("grep", "-r", "-n", "--include=*.rs", "--include=*.go", "--include=*.py", "--include=*.js", "--include=*.ts", "--include=*.md", "--include=*.toml", "--include=*.yaml", "--include=*.json", pattern, searchPath)
	output, _ := cmd.Output()

	result := string(output)
	if len(result) > 20000 {
		result = result[:20000] + "\n... (truncated)"
	}
	return result
}

func fileTree(maxDepth int) string {
	var lines []string

	err := filepath.Walk(workDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		rel, _ := filepath.Rel(workDir, path)
		if rel == "." {
			return nil
		}

		depth := strings.Count(rel, string(os.PathSeparator))
		if depth >= maxDepth {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip hidden and noise
		name := info.Name()
		if strings.HasPrefix(name, ".") || name == "node_modules" || name == "target" || name == "__pycache__" {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		indent := strings.Repeat("  ", depth)
		if info.IsDir() {
			lines = append(lines, fmt.Sprintf("%s%s/", indent, name))
		} else {
			lines = append(lines, fmt.Sprintf("%s%s", indent, name))
		}

		if len(lines) > 300 {
			return fmt.Errorf("limit")
		}
		return nil
	})

	if err != nil && len(lines) == 0 {
		return fmt.Sprintf("Error: %v", err)
	}

	return strings.Join(lines, "\n")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func sendEvent(e Event) {
	data, _ := json.Marshal(e)
	fmt.Println(string(data))
}
