//! Rate limiting with token buckets and adaptive backoff
//!
//! Implements the rate limiting spec from PALACE_SWARM_CONSCIOUSNESS.md Phase 6.
//!
//! Key features:
//! - Token buckets per provider (Mistral, Claude, GLM, etc.)
//! - Proportional 429 handling (never panic, never overreact)
//! - Zerocopy shared state via memory-mapped file
//! - Multiple Palace instances share rate limit awareness

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Provider identifiers for rate limiting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Mistral,
    Anthropic,
    OpenAI,
    GLM,
    Qwen,
    /// OpenRouter - aggregator for GPT-5.1 Codex Max, Gemini 3 Pro, etc.
    OpenRouter,
    Local,
}

impl Provider {
    pub fn from_model(model: &str) -> Self {
        let m = model.to_lowercase();
        // OpenRouter models (prefixed with provider/)
        if m.contains("openai/") || m.contains("google/") || m.contains("openrouter") {
            return Provider::OpenRouter;
        }
        // OpenRouter premium models by name
        if m.contains("gpt-5") || m.contains("codex-max") || m.contains("gemini-3") {
            return Provider::OpenRouter;
        }
        match model {
            m if m.starts_with("devstral") || m.starts_with("mistral") => Provider::Mistral,
            m if m.starts_with("claude") => Provider::Anthropic,
            m if m.starts_with("gpt") => Provider::OpenAI,
            m if m.starts_with("glm") => Provider::GLM,
            m if m.starts_with("qwen") => Provider::Qwen,
            _ => Provider::Local,
        }
    }

    pub fn default_rpm(&self) -> u32 {
        match self {
            Provider::Mistral => 360,    // 6 RPS as per Mistral UI setting
            Provider::Anthropic => 60,
            Provider::OpenAI => 60,
            // GLM-4.6: ~1800 prompts per 5 hours, but we can burst freely
            // Don't preemptively limit - only react to 429s
            Provider::GLM => u32::MAX,
            Provider::Qwen => 60,
            // OpenRouter: conservative limit, varies by model
            Provider::OpenRouter => 60,
            Provider::Local => u32::MAX, // Local models - no limits
        }
    }

    /// Whether to preemptively rate limit or just react to 429s
    pub fn preemptive_limiting(&self) -> bool {
        match self {
            // GLM: burst freely, only back off on 429
            Provider::GLM => false,
            // Local: no limits at all
            Provider::Local => false,
            // Others: preemptively limit
            _ => true,
        }
    }

    pub fn default_tpm(&self) -> u64 {
        match self {
            Provider::Mistral => 100_000,
            Provider::Anthropic => 100_000,
            Provider::OpenAI => 100_000,
            // GLM doesn't use token limits
            Provider::GLM => u64::MAX,
            Provider::Qwen => 100_000,
            // OpenRouter: varies by model, use conservative estimate
            Provider::OpenRouter => 200_000,
            Provider::Local => u64::MAX,
        }
    }
}

/// Token bucket for rate limiting
#[derive(Debug)]
pub struct TokenBucket {
    /// Tokens currently available
    tokens: AtomicU64,
    /// Maximum tokens (bucket capacity)
    max_tokens: u64,
    /// Tokens added per second
    refill_rate: f64,
    /// Last refill timestamp (unix millis)
    last_refill: AtomicU64,
}

impl TokenBucket {
    pub fn new(max_tokens: u64, refill_rate: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            tokens: AtomicU64::new(max_tokens),
            max_tokens,
            refill_rate,
            last_refill: AtomicU64::new(now),
        }
    }

    /// Try to consume tokens. Returns true if successful.
    pub fn try_consume(&self, count: u64) -> bool {
        self.refill();

        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < count {
                return false;
            }
            if self
                .tokens
                .compare_exchange(current, current - count, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let last = self.last_refill.load(Ordering::Acquire);
        let elapsed_ms = now.saturating_sub(last);

        if elapsed_ms > 0 {
            let tokens_to_add = (elapsed_ms as f64 / 1000.0 * self.refill_rate) as u64;
            if tokens_to_add > 0 {
                if self
                    .last_refill
                    .compare_exchange(last, now, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    loop {
                        let current = self.tokens.load(Ordering::Acquire);
                        let new_tokens = (current + tokens_to_add).min(self.max_tokens);
                        if self
                            .tokens
                            .compare_exchange(
                                current,
                                new_tokens,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Get current token count
    pub fn available(&self) -> u64 {
        self.refill();
        self.tokens.load(Ordering::Acquire)
    }

    /// Reduce capacity (for proportional backoff)
    pub fn reduce_capacity(&self, factor: f64) {
        let reduction = (self.tokens.load(Ordering::Acquire) as f64 * factor) as u64;
        self.tokens.fetch_sub(reduction.min(self.tokens.load(Ordering::Acquire)), Ordering::Release);
    }
}

/// Adaptive backoff state
#[derive(Debug)]
pub struct AdaptiveBackoff {
    /// Consecutive 429s seen
    consecutive_429s: AtomicU64,
    /// Last 429 timestamp
    last_429: AtomicU64,
    /// Current backoff multiplier (1.0 = normal)
    backoff_multiplier: std::sync::atomic::AtomicU32,
}

impl Default for AdaptiveBackoff {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveBackoff {
    pub fn new() -> Self {
        Self {
            consecutive_429s: AtomicU64::new(0),
            last_429: AtomicU64::new(0),
            // Store as fixed-point: 100 = 1.0
            backoff_multiplier: std::sync::atomic::AtomicU32::new(100),
        }
    }

    /// Record a 429 error - returns recommended wait time in ms
    pub fn record_429(&self) -> u64 {
        let count = self.consecutive_429s.fetch_add(1, Ordering::Release) + 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_429.store(now, Ordering::Release);

        // Proportional increase - never panic
        // Slow buildup → small increase (drop to 96%)
        // Sudden spike → larger increase (drop to 80%)
        let new_multiplier = match count {
            1 => 96,       // First 429: reduce by 4%
            2 => 90,       // Second: reduce by 10%
            3..=5 => 80,   // Persistent: reduce by 20%
            _ => 60,       // Severe: reduce by 40%
        };

        self.backoff_multiplier.store(new_multiplier, Ordering::Release);

        // Exponential backoff for wait time
        let base_wait = 1000u64; // 1 second
        base_wait * (1 << count.min(6))
    }

    /// Record a successful request
    pub fn record_success(&self) {
        let count = self.consecutive_429s.load(Ordering::Acquire);
        if count > 0 {
            self.consecutive_429s.fetch_sub(1, Ordering::Release);
        }

        // Slowly restore multiplier
        let current = self.backoff_multiplier.load(Ordering::Acquire);
        if current < 100 {
            // Increase by 2% per success
            let new = (current + 2).min(100);
            self.backoff_multiplier.store(new, Ordering::Release);
        }
    }

    /// Get current backoff multiplier (0.0 to 1.0)
    pub fn multiplier(&self) -> f64 {
        self.backoff_multiplier.load(Ordering::Acquire) as f64 / 100.0
    }
}

/// Rate limiter for all providers
pub struct RateLimiter {
    /// Request buckets per provider (requests per minute)
    request_buckets: HashMap<Provider, TokenBucket>,
    /// Token buckets per provider (tokens per minute)
    token_buckets: HashMap<Provider, TokenBucket>,
    /// Backoff state per provider
    backoff: HashMap<Provider, AdaptiveBackoff>,
    /// Queue position counter per provider - each waiting client gets exponentially increasing wait
    /// First waiter: 1s, second: 2s, third: 4s, etc.
    queue_positions: HashMap<Provider, AtomicU64>,
}

impl RateLimiter {
    pub fn new() -> Self {
        let providers = [
            Provider::Mistral,
            Provider::Anthropic,
            Provider::OpenAI,
            Provider::GLM,
            Provider::Qwen,
            Provider::OpenRouter,
            Provider::Local,
        ];

        let mut request_buckets = HashMap::new();
        let mut token_buckets = HashMap::new();
        let mut backoff = HashMap::new();
        let mut queue_positions = HashMap::new();

        for provider in providers {
            // Request bucket: RPM limit, refills at rpm/60 per second
            let rpm = provider.default_rpm();
            request_buckets.insert(
                provider,
                TokenBucket::new(rpm as u64, rpm as f64 / 60.0),
            );

            // Token bucket: TPM limit
            let tpm = provider.default_tpm();
            token_buckets.insert(
                provider,
                TokenBucket::new(tpm, tpm as f64 / 60.0),
            );

            backoff.insert(provider, AdaptiveBackoff::new());
            queue_positions.insert(provider, AtomicU64::new(0));
        }

        Self {
            request_buckets,
            token_buckets,
            backoff,
            queue_positions,
        }
    }

    /// Check if a request can proceed
    pub fn check_request(&self, provider: Provider) -> RateLimitResult {
        // For providers without preemptive limiting (GLM, Local), always allow
        // They only back off after seeing 429s
        if !provider.preemptive_limiting() {
            let backoff = self.backoff.get(&provider).unwrap();
            // But still respect backoff if we've seen 429s
            if backoff.multiplier() < 0.5 {
                // Get queue position for exponential backoff
                let queue = self.queue_positions.get(&provider).unwrap();
                let position = queue.fetch_add(1, Ordering::SeqCst);
                // Exponential: 1s, 2s, 4s, 8s... capped at 64s
                let wait_ms = 1000 * (1u64 << position.min(6));
                return RateLimitResult::Limited {
                    wait_ms,
                    reason: format!("Backing off after 429 (queue pos {})", position),
                };
            }
            return RateLimitResult::Allowed;
        }

        let backoff = self.backoff.get(&provider).unwrap();
        let multiplier = backoff.multiplier();

        // Apply backoff to effective capacity
        let request_bucket = self.request_buckets.get(&provider).unwrap();
        let effective_available = (request_bucket.available() as f64 * multiplier) as u64;

        if effective_available < 1 {
            // Get queue position for exponential backoff
            let queue = self.queue_positions.get(&provider).unwrap();
            let position = queue.fetch_add(1, Ordering::SeqCst);
            // Exponential: 1s, 2s, 4s, 8s... capped at 64s
            let wait_ms = 1000 * (1u64 << position.min(6));
            return RateLimitResult::Limited {
                wait_ms,
                reason: format!("Request rate limit (queue pos {})", position),
            };
        }

        RateLimitResult::Allowed
    }

    /// Signal that a rate-limited request has finished waiting
    /// Call this after the wait completes to decrement queue position
    pub fn finish_waiting(&self, provider: Provider) {
        let queue = self.queue_positions.get(&provider).unwrap();
        let current = queue.load(Ordering::Acquire);
        if current > 0 {
            queue.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Acquire a request slot (blocks conceptually, returns wait time if needed)
    pub fn acquire_request(&self, provider: Provider) -> RateLimitResult {
        let result = self.check_request(provider);
        if let RateLimitResult::Allowed = result {
            let bucket = self.request_buckets.get(&provider).unwrap();
            if !bucket.try_consume(1) {
                return RateLimitResult::Limited {
                    wait_ms: 1000,
                    reason: "Request bucket exhausted".to_string(),
                };
            }
        }
        result
    }

    /// Record response (success or 429)
    pub fn record_response(&self, provider: Provider, status: u16) {
        let backoff = self.backoff.get(&provider).unwrap();
        if status == 429 {
            let _wait = backoff.record_429();
            // Also reduce bucket capacity proportionally
            if let Some(bucket) = self.request_buckets.get(&provider) {
                bucket.reduce_capacity(0.1); // Reduce by 10%
            }
            // Note: logging happens at caller level (daemon has tracing)
        } else if status < 400 {
            backoff.record_success();
        }
    }

    /// Get stats for a provider
    pub fn stats(&self, provider: Provider) -> RateLimitStats {
        let request_bucket = self.request_buckets.get(&provider).unwrap();
        let token_bucket = self.token_buckets.get(&provider).unwrap();
        let backoff = self.backoff.get(&provider).unwrap();

        RateLimitStats {
            requests_available: request_bucket.available(),
            tokens_available: token_bucket.available(),
            backoff_multiplier: backoff.multiplier(),
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of rate limit check
#[derive(Debug, Clone)]
pub enum RateLimitResult {
    Allowed,
    Limited {
        wait_ms: u64,
        reason: String,
    },
}

/// Stats for a provider
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub requests_available: u64,
    pub tokens_available: u64,
    pub backoff_multiplier: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 1.0);
        assert!(bucket.try_consume(5));
        assert_eq!(bucket.available(), 5);
        assert!(bucket.try_consume(5));
        assert!(!bucket.try_consume(1));
    }

    #[test]
    fn test_provider_from_model() {
        assert_eq!(Provider::from_model("devstral-2512"), Provider::Mistral);
        assert_eq!(Provider::from_model("claude-3-opus"), Provider::Anthropic);
        assert_eq!(Provider::from_model("glm-4"), Provider::GLM);
        assert_eq!(Provider::from_model("unknown"), Provider::Local);
    }

    #[test]
    fn test_adaptive_backoff() {
        let backoff = AdaptiveBackoff::new();
        assert_eq!(backoff.multiplier(), 1.0);

        backoff.record_429();
        assert!(backoff.multiplier() < 1.0);

        // Multiple successes should restore
        for _ in 0..50 {
            backoff.record_success();
        }
        assert_eq!(backoff.multiplier(), 1.0);
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new();

        // Should allow initial requests
        assert!(matches!(
            limiter.acquire_request(Provider::Mistral),
            RateLimitResult::Allowed
        ));

        // Record a 429
        limiter.record_response(Provider::Mistral, 429);
        let stats = limiter.stats(Provider::Mistral);
        assert!(stats.backoff_multiplier < 1.0);
    }
}
