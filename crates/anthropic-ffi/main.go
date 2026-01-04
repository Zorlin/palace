package main

/*
#include <stdlib.h>

// Callback type for streaming chunks
typedef void (*stream_callback)(const char* chunk, int is_done, const char* error);

// Helper to invoke the callback from Go
static inline void invoke_callback(stream_callback cb, const char* chunk, int is_done, const char* error) {
    cb(chunk, is_done, error);
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
	if !clientInitialized {
		errStr := C.CString("client not initialized, call anthropic_init first")
		C.invoke_callback(callback, nil, 1, errStr)
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

	stream := client.Messages.NewStreaming(context.Background(), params)

	for stream.Next() {
		event := stream.Current()

		switch eventVariant := event.AsAny().(type) {
		case anthropic.ContentBlockDeltaEvent:
			switch deltaVariant := eventVariant.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				if deltaVariant.Text != "" {
					chunkStr := C.CString(deltaVariant.Text)
					C.invoke_callback(callback, chunkStr, 0, nil)
					C.free(unsafe.Pointer(chunkStr))
				}
			}
		}
	}

	if stream.Err() != nil {
		errStr := C.CString(stream.Err().Error())
		C.invoke_callback(callback, nil, 1, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}

	// Signal completion
	C.invoke_callback(callback, nil, 1, nil)
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
		C.invoke_callback(callback, nil, 1, errStr)
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
		C.invoke_callback(callback, nil, 1, errStr)
		C.free(unsafe.Pointer(errStr))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		errStr := C.CString(string(body))
		C.invoke_callback(callback, nil, 1, errStr)
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
				C.invoke_callback(callback, chunkStr, 0, nil)
				C.free(unsafe.Pointer(chunkStr))
			}
		}
	}

	// Signal completion
	C.invoke_callback(callback, nil, 1, nil)
}

//export anthropic_free_string
func anthropic_free_string(s *C.char) {
	C.free(unsafe.Pointer(s))
}

func main() {}
