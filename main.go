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
	r := gin.Default()
	r.Use(errorHandler())
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
		respData, err = callGemini(req)
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

	systemInstruction := map[string]any{
		"parts": []map[string]string{},
	}
	contents := []map[string]any{}
	for _, msg := range req.Messages {
		role := msg.Role
		if role == "system" {
			// Gemini doesn't support system role
			systemInstruction["parts"] = []map[string]string{
				{"text": msg.Content},
			}
			continue
		}
		contents = append(contents, map[string]any{
			"role": role,
			"parts": []map[string]string{
				{"text": msg.Content},
			},
		})
	}
	body := map[string]any{
		"contents": contents,
	}

	if systemInstruction["parts"] != nil {
		body["systemInstruction"] = systemInstruction
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}

	if req.ResponseType == "text" {
		body["generationConfig"] = map[string]any{
			"responseMimeType": "text/plain",
		}
	} else if req.ResponseType == "json" {
		body["generationConfig"] = map[string]any{
			"responseMimeType": "application/json",
			"responseSchema":   req.ResponseSchema,
		}
	}

	return makeRequest(url, req.APIKey, body, req.Provider)
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
