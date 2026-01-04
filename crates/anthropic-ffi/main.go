package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
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

//export anthropic_free_string
func anthropic_free_string(s *C.char) {
	C.free(unsafe.Pointer(s))
}

func main() {}
