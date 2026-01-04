package main

/*
#include <stdlib.h>

// Event types for streaming
#define EVENT_TEXT 0
#define EVENT_TOOL_USE_START 1
#define EVENT_TOOL_USE_INPUT 2
#define EVENT_TOOL_USE_END 3
#define EVENT_THINKING 4
#define EVENT_DONE 5
#define EVENT_ERROR 6
#define EVENT_TOOL_RESULT 7

// Callback type for streaming events
// event_type: one of EVENT_* constants
// data: event-specific data (text chunk, tool name, JSON input, etc.)
typedef void (*stream_callback)(int event_type, const char* data);

// Tool executor callback - returns tool result as C string (caller must free)
// tool_name: name of tool to execute
// tool_input: JSON input for the tool
typedef char* (*tool_executor)(const char* tool_name, const char* tool_input);

// Helper to invoke the callback from Go
static inline void invoke_callback(stream_callback cb, int event_type, const char* data) {
    cb(event_type, data);
}

// Helper to invoke tool executor from Go
static inline char* invoke_tool(tool_executor exec, const char* name, const char* input) {
    return exec(name, input);
}
*/
import "C"
import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"unsafe"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Response wraps the API response for FFI
type Response struct {
	Success bool   `json:"success"`
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
	Model   string `json:"model"`
	Usage   struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

type apiType int

const (
	apiAnthropic apiType = iota
	apiOpenAI
)

var client anthropic.Client
var clientInitialized bool
var currentAPIType apiType
var openaiKey string
var openaiBase string

//export anthropic_init
func anthropic_init(apiKey *C.char) C.int {
	return anthropic_init_with_base(apiKey, nil)
}

//export anthropic_init_with_base
func anthropic_init_with_base(apiKey *C.char, baseURL *C.char) C.int {
	key := C.GoString(apiKey)
	base := ""
	if baseURL != nil {
		base = C.GoString(baseURL)
	}

	// If key provided, use it directly
	if key != "" {
		opts := []option.RequestOption{option.WithAPIKey(key)}
		if base != "" {
			opts = append(opts, option.WithBaseURL(base))
		}
		client = anthropic.NewClient(opts...)
		clientInitialized = true
		currentAPIType = apiAnthropic
		return 0
	}

	// Try Anthropic from env
	key = os.Getenv("ANTHROPIC_API_KEY")
	if key != "" {
		base = os.Getenv("ANTHROPIC_BASE_URL")
		opts := []option.RequestOption{option.WithAPIKey(key)}
		if base != "" {
			opts = append(opts, option.WithBaseURL(base))
		}
		client = anthropic.NewClient(opts...)
		clientInitialized = true
		currentAPIType = apiAnthropic
		return 0
	}

	// Try Z.AI from env
	key = os.Getenv("ZAI_API_KEY")
	if key != "" {
		base = os.Getenv("ZAI_BASE_URL")
		if base == "" {
			base = "https://api.z.ai/api/anthropic"
		}
		client = anthropic.NewClient(
			option.WithAPIKey(key),
			option.WithBaseURL(base),
		)
		clientInitialized = true
		currentAPIType = apiAnthropic
		return 0
	}

	// Try generic OpenAI from env
	key = os.Getenv("OPENAI_API_KEY")
	if key != "" {
		openaiKey = key
		openaiBase = os.Getenv("OPENAI_API_BASE")
		if openaiBase == "" {
			openaiBase = "https://api.openai.com/v1"
		}
		clientInitialized = true
		currentAPIType = apiOpenAI
		return 0
	}

	return -1
}

//export anthropic_message
func anthropic_message(model *C.char, systemPrompt *C.char, userMessage *C.char, maxTokens C.int) *C.char {
	if !clientInitialized {
		resp := Response{Success: false, Error: "client not initialized, call anthropic_init first"}
		jsonBytes, _ := json.Marshal(resp)
		return C.CString(string(jsonBytes))
	}

	modelStr := C.GoString(model)
	systemStr := C.GoString(systemPrompt)
	userStr := C.GoString(userMessage)
	tokens := int64(maxTokens)

	if tokens <= 0 {
		tokens = 4096
	}

	if currentAPIType == apiOpenAI {
		return callOpenAI(modelStr, systemStr, userStr, tokens)
	}

	// Anthropic API
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(modelStr),
		MaxTokens: tokens,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userStr)),
		},
	}

	if systemStr != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemStr},
		}
	}

	message, err := client.Messages.New(context.Background(), params)
	if err != nil {
		resp := Response{Success: false, Error: err.Error()}
		jsonBytes, _ := json.Marshal(resp)
		return C.CString(string(jsonBytes))
	}

	// Extract text content
	var content string
	for _, block := range message.Content {
		if block.Type == "text" {
			content += block.Text
		}
	}

	resp := Response{
		Success: true,
		Content: content,
		Model:   string(message.Model),
	}
	resp.Usage.InputTokens = message.Usage.InputTokens
	resp.Usage.OutputTokens = message.Usage.OutputTokens

	jsonBytes, _ := json.Marshal(resp)
	return C.CString(string(jsonBytes))
}

// OpenAI-compatible API call
func callOpenAI(model, system, user string, maxTokens int64) *C.char {
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type Request struct {
		Model     string    `json:"model"`
		Messages  []Message `json:"messages"`
		MaxTokens int64     `json:"max_tokens"`
	}
	type Choice struct {
		Message Message `json:"message"`
	}
	type Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	}
	type OpenAIResponse struct {
		Choices []Choice `json:"choices"`
		Model   string   `json:"model"`
		Usage   Usage    `json:"usage"`
	}

	messages := []Message{}
	if system != "" {
		messages = append(messages, Message{Role: "system", Content: system})
	}
	messages = append(messages, Message{Role: "user", Content: user})

	reqBody := Request{
		Model:     model,
		Messages:  messages,
		MaxTokens: maxTokens,
	}

	jsonBody, _ := json.Marshal(reqBody)
	req, err := http.NewRequest("POST", openaiBase+"/chat/completions", bytes.NewBuffer(jsonBody))
	if err != nil {
		resp := Response{Success: false, Error: err.Error()}
		jsonBytes, _ := json.Marshal(resp)
		return C.CString(string(jsonBytes))
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+openaiKey)

	httpClient := &http.Client{}
	httpResp, err := httpClient.Do(req)
	if err != nil {
		resp := Response{Success: false, Error: err.Error()}
		jsonBytes, _ := json.Marshal(resp)
		return C.CString(string(jsonBytes))
	}
	defer httpResp.Body.Close()

	body, _ := io.ReadAll(httpResp.Body)

	if httpResp.StatusCode != 200 {
		resp := Response{Success: false, Error: string(body)}
		jsonBytes, _ := json.Marshal(resp)
		return C.CString(string(jsonBytes))
	}

	var openaiResp OpenAIResponse
	if err := json.Unmarshal(body, &openaiResp); err != nil {
		resp := Response{Success: false, Error: "Failed to parse response: " + err.Error()}
		jsonBytes, _ := json.Marshal(resp)
		return C.CString(string(jsonBytes))
	}

	content := ""
	if len(openaiResp.Choices) > 0 {
		content = openaiResp.Choices[0].Message.Content
	}

	resp := Response{
		Success: true,
		Content: content,
		Model:   openaiResp.Model,
	}
	resp.Usage.InputTokens = openaiResp.Usage.PromptTokens
	resp.Usage.OutputTokens = openaiResp.Usage.CompletionTokens

	jsonBytes, _ := json.Marshal(resp)
	return C.CString(string(jsonBytes))
}

//export anthropic_message_stream
func anthropic_message_stream(model *C.char, systemPrompt *C.char, userMessage *C.char, maxTokens C.int, callback C.stream_callback) {
	anthropic_message_stream_with_tools(model, systemPrompt, userMessage, maxTokens, nil, callback)
}

//export anthropic_message_stream_with_tools
func anthropic_message_stream_with_tools(model *C.char, systemPrompt *C.char, userMessage *C.char, maxTokens C.int, toolsJSON *C.char, callback C.stream_callback) {
	if !clientInitialized {
		errStr := C.CString("client not initialized, call anthropic_init first")
		C.invoke_callback(callback, C.EVENT_ERROR, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}

	modelStr := C.GoString(model)
	systemStr := C.GoString(systemPrompt)
	userStr := C.GoString(userMessage)
	tokens := int64(maxTokens)

	if tokens <= 0 {
		tokens = 4096
	}

	if currentAPIType == apiOpenAI {
		streamOpenAI(modelStr, systemStr, userStr, tokens, callback)
		return
	}

	// Anthropic streaming API
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(modelStr),
		MaxTokens: tokens,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userStr)),
		},
	}

	if systemStr != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: systemStr},
		}
	}

	// Add tools if provided
	if toolsJSON != nil {
		toolsStr := C.GoString(toolsJSON)
		if toolsStr != "" {
			var tools []anthropic.ToolUnionParam
			if err := json.Unmarshal([]byte(toolsStr), &tools); err == nil {
				params.Tools = tools
			}
		}
	}

	stream := client.Messages.NewStreaming(context.Background(), params)

	for stream.Next() {
		event := stream.Current()

		switch eventVariant := event.AsAny().(type) {
		case anthropic.ContentBlockStartEvent:
			// Detect tool use or thinking block starting
			switch blockVariant := eventVariant.ContentBlock.AsAny().(type) {
			case anthropic.ToolUseBlock:
				// Tool use starting - emit tool name
				toolInfo := blockVariant.Name
				if blockVariant.ID != "" {
					toolInfo = blockVariant.ID + ":" + blockVariant.Name
				}
				infoStr := C.CString(toolInfo)
				C.invoke_callback(callback, C.EVENT_TOOL_USE_START, infoStr)
				C.free(unsafe.Pointer(infoStr))
			case anthropic.ThinkingBlock:
				// Thinking block starting
				if blockVariant.Thinking != "" {
					thinkStr := C.CString(blockVariant.Thinking)
					C.invoke_callback(callback, C.EVENT_THINKING, thinkStr)
					C.free(unsafe.Pointer(thinkStr))
				}
			}

		case anthropic.ContentBlockDeltaEvent:
			switch deltaVariant := eventVariant.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				if deltaVariant.Text != "" {
					chunkStr := C.CString(deltaVariant.Text)
					C.invoke_callback(callback, C.EVENT_TEXT, chunkStr)
					C.free(unsafe.Pointer(chunkStr))
				}
			case anthropic.InputJSONDelta:
				// Tool input JSON streaming
				if deltaVariant.PartialJSON != "" {
					jsonStr := C.CString(deltaVariant.PartialJSON)
					C.invoke_callback(callback, C.EVENT_TOOL_USE_INPUT, jsonStr)
					C.free(unsafe.Pointer(jsonStr))
				}
			case anthropic.ThinkingDelta:
				// Thinking text streaming
				if deltaVariant.Thinking != "" {
					thinkStr := C.CString(deltaVariant.Thinking)
					C.invoke_callback(callback, C.EVENT_THINKING, thinkStr)
					C.free(unsafe.Pointer(thinkStr))
				}
			}

		case anthropic.ContentBlockStopEvent:
			// Content block ended - could signal tool use end
			C.invoke_callback(callback, C.EVENT_TOOL_USE_END, nil)
		}
	}

	if stream.Err() != nil {
		errStr := C.CString(stream.Err().Error())
		C.invoke_callback(callback, C.EVENT_ERROR, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}

	// Signal completion
	C.invoke_callback(callback, C.EVENT_DONE, nil)
}

// OpenAI-compatible streaming API call
func streamOpenAI(model, system, user string, maxTokens int64, callback C.stream_callback) {
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type Request struct {
		Model     string    `json:"model"`
		Messages  []Message `json:"messages"`
		MaxTokens int64     `json:"max_tokens"`
		Stream    bool      `json:"stream"`
	}

	messages := []Message{}
	if system != "" {
		messages = append(messages, Message{Role: "system", Content: system})
	}
	messages = append(messages, Message{Role: "user", Content: user})

	reqBody := Request{
		Model:     model,
		Messages:  messages,
		MaxTokens: maxTokens,
		Stream:    true,
	}

	jsonBody, _ := json.Marshal(reqBody)
	req, err := http.NewRequest("POST", openaiBase+"/chat/completions", bytes.NewBuffer(jsonBody))
	if err != nil {
		errStr := C.CString(err.Error())
		C.invoke_callback(callback, C.EVENT_ERROR, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+openaiKey)
	req.Header.Set("Accept", "text/event-stream")

	httpClient := &http.Client{}
	resp, err := httpClient.Do(req)
	if err != nil {
		errStr := C.CString(err.Error())
		C.invoke_callback(callback, C.EVENT_ERROR, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		errStr := C.CString(string(body))
		C.invoke_callback(callback, C.EVENT_ERROR, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}

	// Parse SSE stream
	type Delta struct {
		Content string `json:"content"`
	}
	type Choice struct {
		Delta Delta `json:"delta"`
	}
	type StreamChunk struct {
		Choices []Choice `json:"choices"`
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk StreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				chunkStr := C.CString(chunk.Choices[0].Delta.Content)
				C.invoke_callback(callback, C.EVENT_TEXT, chunkStr)
				C.free(unsafe.Pointer(chunkStr))
			}
		}
	}

	// Signal completion
	C.invoke_callback(callback, C.EVENT_DONE, nil)
}

//export anthropic_agentic_loop
func anthropic_agentic_loop(model *C.char, systemPrompt *C.char, userMessage *C.char, maxTokens C.int, toolsJSON *C.char, callback C.stream_callback, toolExecutor C.tool_executor) {
	if !clientInitialized {
		errStr := C.CString("client not initialized")
		C.invoke_callback(callback, C.EVENT_ERROR, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}

	modelStr := C.GoString(model)
	systemStr := C.GoString(systemPrompt)
	userStr := C.GoString(userMessage)
	tokens := int64(maxTokens)
	if tokens <= 0 {
		tokens = 8192
	}

	// Parse tools
	var tools []anthropic.ToolUnionParam
	if toolsJSON != nil {
		toolsStr := C.GoString(toolsJSON)
		if toolsStr != "" {
			json.Unmarshal([]byte(toolsStr), &tools)
		}
	}

	// Build initial messages
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(userStr)),
	}

	// Agentic loop - continue until no more tool calls
	for {
		params := anthropic.MessageNewParams{
			Model:     anthropic.Model(modelStr),
			MaxTokens: tokens,
			Messages:  messages,
		}
		if systemStr != "" {
			params.System = []anthropic.TextBlockParam{{Text: systemStr}}
		}
		if len(tools) > 0 {
			params.Tools = tools
		}

		// Stream this turn
		stream := client.Messages.NewStreaming(context.Background(), params)

		var toolCalls []struct {
			ID    string
			Name  string
			Input string
		}
		var currentToolID, currentToolName string
		var currentToolInput strings.Builder
		var textContent strings.Builder

		for stream.Next() {
			event := stream.Current()

			switch ev := event.AsAny().(type) {
			case anthropic.ContentBlockStartEvent:
				switch block := ev.ContentBlock.AsAny().(type) {
				case anthropic.ToolUseBlock:
					currentToolID = block.ID
					currentToolName = block.Name
					currentToolInput.Reset()
					// Emit tool start
					info := C.CString(currentToolID + ":" + currentToolName)
					C.invoke_callback(callback, C.EVENT_TOOL_USE_START, info)
					C.free(unsafe.Pointer(info))
				}

			case anthropic.ContentBlockDeltaEvent:
				switch delta := ev.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if delta.Text != "" {
						textContent.WriteString(delta.Text)
						chunk := C.CString(delta.Text)
						C.invoke_callback(callback, C.EVENT_TEXT, chunk)
						C.free(unsafe.Pointer(chunk))
					}
				case anthropic.InputJSONDelta:
					if delta.PartialJSON != "" {
						currentToolInput.WriteString(delta.PartialJSON)
						chunk := C.CString(delta.PartialJSON)
						C.invoke_callback(callback, C.EVENT_TOOL_USE_INPUT, chunk)
						C.free(unsafe.Pointer(chunk))
					}
				case anthropic.ThinkingDelta:
					if delta.Thinking != "" {
						chunk := C.CString(delta.Thinking)
						C.invoke_callback(callback, C.EVENT_THINKING, chunk)
						C.free(unsafe.Pointer(chunk))
					}
				}

			case anthropic.ContentBlockStopEvent:
				if currentToolName != "" {
					// Tool block complete - save it
					toolCalls = append(toolCalls, struct {
						ID    string
						Name  string
						Input string
					}{currentToolID, currentToolName, currentToolInput.String()})
					C.invoke_callback(callback, C.EVENT_TOOL_USE_END, nil)
					currentToolName = ""
				}
			}
		}

		if stream.Err() != nil {
			errStr := C.CString(stream.Err().Error())
			C.invoke_callback(callback, C.EVENT_ERROR, errStr)
			C.free(unsafe.Pointer(errStr))
			return
		}

		// If no tool calls, we're done
		if len(toolCalls) == 0 {
			C.invoke_callback(callback, C.EVENT_DONE, nil)
			return
		}

		// Execute tools and build assistant + user messages
		var assistantContent []anthropic.ContentBlockParamUnion
		var toolResults []anthropic.ContentBlockParamUnion

		// Add text if any
		if textContent.Len() > 0 {
			assistantContent = append(assistantContent, anthropic.NewTextBlock(textContent.String()))
		}

		// Add tool uses and execute them
		for _, tc := range toolCalls {
			// Add tool use to assistant message
			var inputJSON map[string]interface{}
			json.Unmarshal([]byte(tc.Input), &inputJSON)
			assistantContent = append(assistantContent, anthropic.NewToolUseBlock(tc.ID, inputJSON, tc.Name))

			// Execute tool via callback
			nameC := C.CString(tc.Name)
			inputC := C.CString(tc.Input)
			resultC := C.invoke_tool(toolExecutor, nameC, inputC)
			C.free(unsafe.Pointer(nameC))
			C.free(unsafe.Pointer(inputC))

			result := ""
			if resultC != nil {
				result = C.GoString(resultC)
				C.free(unsafe.Pointer(resultC))
			}

			// Emit tool result
			resultInfo := C.CString(tc.Name + ": " + result[:min(100, len(result))])
			C.invoke_callback(callback, C.EVENT_TOOL_RESULT, resultInfo)
			C.free(unsafe.Pointer(resultInfo))

			// Add tool result
			toolResults = append(toolResults, anthropic.NewToolResultBlock(tc.ID, result, false))
		}

		// Add messages for next turn
		messages = append(messages, anthropic.MessageParam{
			Role:    "assistant",
			Content: assistantContent,
		})
		messages = append(messages, anthropic.NewUserMessage(toolResults...))
	}
}

//export anthropic_free_string
func anthropic_free_string(s *C.char) {
	C.free(unsafe.Pointer(s))
}

func main() {}
