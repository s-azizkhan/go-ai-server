package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
	"github.com/sirupsen/logrus"
)

type Request struct {
	Provider       string          `json:"provider" binding:"required,oneof=ollama openai gemini grok"`
	APIKey         string          `json:"api_key" binding:"required"`
	ModelID        string          `json:"model_id" binding:"required"`
	Messages       []Message       `json:"messages" binding:"required,dive"`
	ResponseType   string          `json:"response_type" binding:"oneof=text json"`
	ResponseSchema json.RawMessage `json:"response_schema"`
	CustomURL      string          `json:"custom_url"`
	Tools          []any           `json:"tools"`
	Stream         bool            `json:"stream"`
}

type Message struct {
	Role      string `json:"role" binding:"required,oneof=system user assistant"`
	Content   string `json:"content" binding:"required"`
	ToolCalls []any  `json:"tool_calls"`
}

type Response struct {
	Data  any    `json:"data"`
	Error string `json:"error,omitempty"`
}

type ErrorResponse struct {
	Error            string `json:"error"`
	ProviderResponse string `json:"provider_response"`
}

type APIError struct {
	Message    string `json:"message"`
	StatusCode int    `json:"status_code,omitempty"`
}

// Gemini Response Structures (as defined previously)
type GeminiResponse struct {
	Candidates []struct {
		AvgLogprobs float64 `json:"avgLogprobs"`
		Content     struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
			Role string `json:"role"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	ModelVersion  string `json:"modelVersion"`
	UsageMetadata struct {
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		CandidatesTokensDetails []struct {
			Modality   string `json:"modality"`
			TokenCount int    `json:"tokenCount"`
		} `json:"candidatesTokensDetails"`
		PromptTokenCount    int `json:"promptTokenCount"`
		PromptTokensDetails []struct {
			Modality   string `json:"modality"`
			TokenCount int    `json:"tokenCount"`
		} `json:"promptTokensDetails"`
		TotalTokenCount int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// Ollama Response Structure (as defined previously)
type OllamaResponse struct {
	CreatedAt          string    `json:"created_at"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason"`
	EvalCount          int       `json:"eval_count"`
	EvalDuration       int64     `json:"eval_duration"`
	LoadDuration       int64     `json:"load_duration"`
	Message            AIMessage `json:"message"`
	Model              string    `json:"model"`
	PromptEvalCount    int       `json:"prompt_eval_count"`
	PromptEvalDuration int64     `json:"prompt_eval_duration"`
	TotalDuration      int64     `json:"total_duration"`
}

type AIResponse struct {
	CreatedAt          string    `json:"created_at"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason"`
	EvalCount          int       `json:"eval_count"`
	EvalDuration       int64     `json:"eval_duration"`
	LoadDuration       int64     `json:"load_duration"`
	Message            AIMessage `json:"message"`
	Model              string    `json:"model"`
	PromptEvalCount    int       `json:"prompt_eval_count"`
	PromptEvalDuration int64     `json:"prompt_eval_duration"`
	TotalDuration      int64     `json:"total_duration"`
}

type AIMessage struct {
	Content string `json:"content"`
	Role    string `json:"role"`
}

// ParseGeminiResponse converts Gemini response to standardized AI response format
func ParseGeminiResponse(geminiJSON any) (any, error) {
	var inputResp GeminiResponse
	jsonBytes, err := json.Marshal(geminiJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Gemini response to JSON: %w", err)
	}
	if unmarshalErr := json.Unmarshal(jsonBytes, &inputResp); unmarshalErr != nil {
		return nil, fmt.Errorf("failed to unmarshal Gemini response: %w", err)
	}

	if len(inputResp.Candidates) == 0 || len(inputResp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("Gemini response does not contain valid content")
	}

	// Create output response
	outputResp := AIResponse{
		CreatedAt:       time.Now().UTC().Format(time.RFC3339Nano),
		Done:            true,
		Model:           inputResp.ModelVersion, // Hardcoded as per desired output
		DoneReason:      inputResp.Candidates[0].FinishReason,
		EvalCount:       inputResp.UsageMetadata.CandidatesTokenCount,
		PromptEvalCount: inputResp.UsageMetadata.PromptTokenCount,
		// Assuming reasonable defaults for durations since input doesn't provide them
		EvalDuration:       0, // nanoseconds
		LoadDuration:       0, // nanoseconds
		PromptEvalDuration: 0, // nanoseconds
		TotalDuration:      0, // nanoseconds
		Message: AIMessage{
			Content: inputResp.Candidates[0].Content.Parts[0].Text,
			Role:    "assistant",
		},
	}

	aiResp, err := json.Marshal(outputResp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Ollama response: %w", err)
	}

	// convert bytes to interface
	var aiRespMap map[string]any
	if err := json.Unmarshal(aiResp, &aiRespMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal AI response: %w", err)
	}

	return aiRespMap, nil
}

func (e *APIError) Error() string {
	return fmt.Sprintf(`{"error": %s, "status_code": %d}`, e.Message, e.StatusCode)
}

var validate *validator.Validate
var log = logrus.New()

func init() {
	validate = validator.New()
	log.SetFormatter(&logrus.JSONFormatter{})
	log.SetLevel(logrus.InfoLevel)
}

func main() {
	// gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.Use(errorHandler())
	r.GET("/healthz", handleHealthCheck)
	r.POST("/api/chat", handleChat)
	if err := r.Run(":4040"); err != nil {
		panic(fmt.Sprintf("Failed to start server: %v", err))
	}
}

func errorHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()
		if len(c.Errors) > 0 {
			c.JSON(http.StatusBadRequest, ErrorResponse{Error: c.Errors.String()})
		}
	}
}

func handleHealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "ok",
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

func handleChat(c *gin.Context) {
	var req Request
	if err := c.ShouldBindJSON(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, ErrorResponse{Error: "Invalid request body: " + err.Error()})
		return
	}

	if req.ResponseType == "" {
		req.ResponseType = "text"
	}

	if req.ResponseType == "json" && len(req.ResponseSchema) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, ErrorResponse{Error: "response_schema required for json response_type"})
		return
	}

	if req.CustomURL != "" {
		if _, err := url.ParseRequestURI(req.CustomURL); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, ErrorResponse{Error: "Invalid custom_url"})
			return
		}
	}

	if len(req.Messages) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, ErrorResponse{Error: "At least one message is required"})
		return
	}

	for _, msg := range req.Messages {
		if err := validate.Struct(msg); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, ErrorResponse{Error: "Invalid message: " + err.Error()})
			return
		}
	}

	var respData any
	var err error

	switch strings.ToLower(req.Provider) {
	case "ollama":
		respData, err = callOllama(req)
	case "openai":
		respData, err = callOpenAI(req)
	case "gemini":
		var geminiRespData any
		geminiRespData, err = callGemini(req)
		// print the data type of geminiRespData
		fmt.Printf("Data type of geminiRespData: %T\n", geminiRespData)
		var ollamaResp any
		ollamaResp, err = ParseGeminiResponse(geminiRespData)
		if err != nil {
			log.Fatalf("failed to convert Gemini response: %d", err)
			// c.AbortWithStatusJSON(http.StatusBadRequest,j ErrorResponse{Error: "failed to convert Gemini response"})
			// return
			respData = geminiRespData
		} else {
			respData = ollamaResp
		}

	case "grok":
		respData, err = callGrok(req)
	default:
		c.AbortWithStatusJSON(http.StatusBadRequest, ErrorResponse{Error: "Unsupported provider"})
		return
	}

	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Provider error: %v", err), ProviderResponse: err.Error()})
		return
	}

	c.JSON(http.StatusOK, Response{Data: respData})
	return
}

func callOllama(req Request) (any, error) {
	url := req.CustomURL
	if url == "" {
		url = "http://127.0.0.1:11434/api/chat"
	}
	body := map[string]any{
		"model":    req.ModelID,
		"messages": req.Messages,
		"stream":   req.Stream,
	}
	if req.ResponseType == "json" && len(req.ResponseSchema) > 0 {
		body["format"] = req.ResponseSchema
		body["stream"] = false
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}

	return makeRequest(url, req.APIKey, body, req.Provider)
}

func callOpenAI(req Request) (any, error) {
	url := req.CustomURL
	if url == "" {
		url = "https://api.openai.com/v1/chat/completions"
	}
	body := map[string]any{
		"model":    req.ModelID,
		"messages": req.Messages,
		"stream":   req.Stream,
	}
	if req.ResponseType == "json" && len(req.ResponseSchema) > 0 {
		body["response_format"] = map[string]any{"type": "json_object"}
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}

	return makeRequest(url, req.APIKey, body, req.Provider)
}

func callGemini(req Request) (any, error) {
	url := req.CustomURL
	if url == "" {
		url = fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", req.ModelID, req.APIKey)
	}

	body := map[string]any{
		"contents": buildGeminiContents(req.Messages),
	}

	// Add system instruction if present
	if systemParts, exists := extractGeminiSystemInstruction(req.Messages); exists {
		body["systemInstruction"] = map[string]any{
			"parts": systemParts,
		}
	}

	// Add tools if provided
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}

	// Configure response type
	if req.ResponseType != "" {
		config := map[string]any{"responseMimeType": "text/plain"}
		if req.ResponseType == "json" {
			config["responseMimeType"] = "application/json"
			config["responseSchema"] = req.ResponseSchema
		}
		body["generationConfig"] = config
	}

	return makeRequest(url, req.APIKey, body, req.Provider)
}

// buildContents constructs the contents array, excluding system messages
func buildGeminiContents(messages []Message) []map[string]any {
	contents := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		if msg.Role == "system" {
			continue
		}
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}
		contents = append(contents, map[string]any{
			"role": role,
			"parts": []map[string]string{
				{"text": msg.Content},
			},
		})
	}
	return contents
}

// extractSystemInstruction extracts system message parts if present
func extractGeminiSystemInstruction(messages []Message) ([]map[string]string, bool) {
	for _, msg := range messages {
		if msg.Role == "system" {
			return []map[string]string{{"text": msg.Content}}, true
		}
	}
	return nil, false
}

func callGrok(req Request) (any, error) {
	url := req.CustomURL
	if url == "" {
		url = "https://api.x.ai/v1/grok"
	}
	body := map[string]any{
		"model":    req.ModelID,
		"messages": req.Messages,
		"stream":   req.Stream,
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}

	return makeRequest(url, req.APIKey, body, req.Provider)
}

func makeRequest(urlStr string, apiKey string, body any, provider string) (any, error) {
	logEntry := log.WithFields(logrus.Fields{
		"url": urlStr,
	})

	jsonBody, err := json.Marshal(body)
	if err != nil {
		logEntry.WithError(err).Error("Failed to marshal request body")
		return nil, &APIError{Message: fmt.Sprintf("failed to marshal request body: %v", err)}
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
	}
	req, err := http.NewRequest("POST", urlStr, bytes.NewBuffer(jsonBody))
	if err != nil {
		logEntry.WithError(err).Error("Failed to create request")
		return nil, &APIError{Message: fmt.Sprintf("failed to create request: %v", err)}
	}

	req.Header.Set("Content-Type", "application/json")
	if provider != "gemini" { // BCOZ: Gemini uses API key in url
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	req.Header.Set("Accept", "application/json")

	logEntry.Info("Sending request to provider")
	resp, err := client.Do(req)
	if err != nil {
		logEntry.WithError(err).Error("Request to provider failed")
		return nil, &APIError{Message: fmt.Sprintf("request failed: %v", err)}
	}
	defer resp.Body.Close()

	logEntry.WithField("status_code", resp.StatusCode).Info("Received response from provider")
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		logEntry.WithFields(logrus.Fields{
			"status_code": resp.StatusCode,
			"response":    string(bodyBytes),
		}).Error("Provider returned non-OK status")
		return nil, &APIError{
			Message:    string(bodyBytes),
			StatusCode: resp.StatusCode,
		}
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		logEntry.WithError(err).Error("Failed to read response body")
		return nil, &APIError{Message: fmt.Sprintf("failed to read response: %v", err)}
	}

	var result map[string]any
	if err := json.Unmarshal(respBody, &result); err != nil {
		logEntry.WithError(err).Error("Failed to parse JSON response")
		return nil, &APIError{Message: fmt.Sprintf("failed to parse JSON response: %v", err)}
	}
	return result, nil
}
