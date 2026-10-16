# X "For You" Feed — Recommendation Algorithm Implementation Analysis

This document provides a detailed technical analysis of the X "For You" feed recommendation algorithm as implemented in this codebase. All findings are based on the source code contained in this repository, with explicit references to specific files, functions, and classes.

---

## Table of Contents

1. [Repository Map](#1-repository-map)
2. [Request Entry Point](#2-request-entry-point)
3. [Stage 0 — Query Hydration](#3-stage-0--query-hydration)
4. [Stage 1 — Candidate Sourcing (Retrieval)](#4-stage-1--candidate-sourcing-retrieval)
   - [In-Network Source: Thunder](#in-network-source-thunder)
   - [Out-of-Network Source: Phoenix Retrieval](#out-of-network-source-phoenix-retrieval)
5. [Stage 2 — Candidate Hydration (Feature Enrichment)](#5-stage-2--candidate-hydration-feature-enrichment)
6. [Stage 3 — Pre-Scoring Filters](#6-stage-3--pre-scoring-filters)
7. [Stage 4 — Scoring and Ranking](#7-stage-4--scoring-and-ranking)
   - [Phoenix Scorer (ML Prediction)](#phoenix-scorer-ml-prediction)
   - [Weighted Scorer (Score Fusion)](#weighted-scorer-score-fusion)
   - [Author Diversity Scorer](#author-diversity-scorer)
   - [OON Scorer (In/Out-of-Network Adjustment)](#oon-scorer-inout-of-network-adjustment)
8. [Stage 5 — Selection](#8-stage-5--selection)
9. [Stage 6 — Post-Selection Processing](#9-stage-6--post-selection-processing)
10. [Stage 7 — Side Effects](#10-stage-7--side-effects)
11. [Phoenix ML Architecture (Deep Dive)](#11-phoenix-ml-architecture-deep-dive)
    - [Ranking Model](#ranking-model)
    - [Retrieval Model (Two-Tower)](#retrieval-model-two-tower)
    - [Shared Transformer (Grok Architecture)](#shared-transformer-grok-architecture)
    - [Input Embeddings and Feature Encoding](#input-embeddings-and-feature-encoding)
    - [Candidate Isolation Attention Mask](#candidate-isolation-attention-mask)
12. [Thunder In-Memory Post Store (Deep Dive)](#12-thunder-in-memory-post-store-deep-dive)
13. [Candidate Pipeline Framework](#13-candidate-pipeline-framework)
14. [Complete Data Flow Summary](#14-complete-data-flow-summary)
15. [Key Engineering Decisions](#15-key-engineering-decisions)

---

## 1. Repository Map

```
x-algorithm/
├── candidate-pipeline/          # Rust: Reusable pipeline execution framework
│   ├── candidate_pipeline.rs    # CandidatePipeline trait & execute() orchestration
│   ├── source.rs                # Source<Q, C> trait
│   ├── hydrator.rs              # Hydrator<Q, C> trait
│   ├── filter.rs                # Filter<Q, C> trait
│   ├── scorer.rs                # Scorer<Q, C> trait
│   ├── selector.rs              # Selector<Q, C> trait
│   └── side_effect.rs           # SideEffect<Q, C> trait
│
├── home-mixer/                  # Rust: Orchestration service for For You feed
│   ├── main.rs                  # gRPC server entry point
│   ├── server.rs                # ScoredPostsService handler
│   ├── candidate_pipeline/
│   │   ├── phoenix_candidate_pipeline.rs  # Full pipeline wiring
│   │   ├── candidate.rs                   # PostCandidate data struct
│   │   ├── query.rs                       # ScoredPostsQuery data struct
│   │   └── candidate_features.rs          # Feature constants
│   ├── query_hydrators/         # Fetch user context (history, following list)
│   ├── sources/                 # Thunder (in-net) + Phoenix (out-of-net)
│   ├── candidate_hydrators/     # Enrich candidates with metadata
│   ├── filters/                 # Remove ineligible candidates
│   ├── scorers/                 # ML scoring + weighted combination + diversity
│   ├── selectors/               # Sort and pick top-K
│   └── side_effects/            # Async post-pipeline effects (caching)
│
├── phoenix/                     # Python/JAX: ML models
│   ├── grok.py                  # Grok-based Transformer architecture
│   ├── recsys_model.py          # PhoenixModel (ranking)
│   ├── recsys_retrieval_model.py # PhoenixRetrievalModel (two-tower retrieval)
│   ├── runners.py               # Inference runners + batch utilities
│   ├── run_ranker.py            # Ranking demo script
│   └── run_retrieval.py         # Retrieval demo script
│
└── thunder/                     # Rust: In-memory in-network post store
    ├── main.rs                  # Kafka consumer + gRPC server
    ├── thunder_service.rs       # InNetworkPostsService handler
    └── posts/                   # PostStore (per-user in-memory store)
```

---

## 2. Request Entry Point

**File:** `home-mixer/main.rs`
**Server class:** `HomeMixerServer` (`home-mixer/server.rs`)

The system exposes a gRPC endpoint: `ScoredPostsService.GetScoredPosts`.

```
gRPC Request (ScoredPostsQuery)
     viewer_id, client_app_id, country_code, language_code,
     seen_ids, served_ids, in_network_only, is_bottom_request,
     bloom_filter_entries
         │
         ▼
  server.rs: HomeMixerServer::get_scored_posts()
         │
         ▼
  Creates ScoredPostsQuery
         │
         ▼
  phx_candidate_pipeline.execute(query)
         │
         ▼
  gRPC Response (ScoredPostsResponse)
     → Vec<ScoredPost> { tweet_id, score, in_network, ... }
```

**Key observation:** The server validates that `viewer_id != 0`, then converts the proto request into a `ScoredPostsQuery` and delegates all logic to the `PhoenixCandidatePipeline`.

---

## 3. Stage 0 — Query Hydration

**File:** `home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs`
**Executed by:** `candidate-pipeline/candidate_pipeline.rs:hydrate_query()` (parallel)

Before any candidates are fetched, the query is enriched with user context that will be needed by downstream components.

| Component | File | What It Fetches |
|---|---|---|
| `UserActionSeqQueryHydrator` | `query_hydrators/user_action_seq_query_hydrator.rs` | User's engagement history (UAS — User Action Sequence): recent likes, replies, reposts, clicks, etc. |
| `UserFeaturesQueryHydrator` | `query_hydrators/user_features_query_hydrator.rs` | User's social graph features: list of followed user IDs, muted keywords, blocked users |

The two hydrators run **in parallel** (`join_all` in `candidate_pipeline.rs:hydrate_query()`). Their outputs are merged into the `ScoredPostsQuery` object, which is then passed immutably to all downstream stages.

---

## 4. Stage 1 — Candidate Sourcing (Retrieval)

**Executed by:** `candidate_pipeline.rs:fetch_candidates()` (parallel)

Both sources run **in parallel**. Results are concatenated into a single flat list.

### In-Network Source: Thunder

**File:** `home-mixer/sources/thunder_source.rs`
**Service:** Thunder gRPC `InNetworkPostsServiceClient::get_in_network_posts()`

Thunder is queried with:
- `user_id` — the viewer
- `following_user_ids` — from `query.user_features.followed_user_ids`
- `max_results` — configured via `params::THUNDER_MAX_RESULTS`

Thunder returns recent posts from accounts the user follows. Each post becomes a `PostCandidate` with `served_type = ForYouInNetwork`.

**Thunder Architecture** (see `thunder/main.rs`):
- Consumes real-time post create/delete events from Kafka
- Maintains a `PostStore` — an in-memory, per-user index of recent posts
- Automatically trims posts older than the retention period (configurable, runs every 2 minutes)
- On startup, waits for Kafka catchup before declaring readiness

### Out-of-Network Source: Phoenix Retrieval

**File:** `home-mixer/sources/phoenix_source.rs`
**Gated by:** `enable()` returns `false` when `query.in_network_only = true`

Phoenix Retrieval is called with:
- `user_id`
- `user_action_sequence` — the engagement history fetched by `UserActionSeqQueryHydrator`
- `max_results` — configured via `params::PHOENIX_MAX_RESULTS`

The Phoenix retrieval service runs the **two-tower ML model** (see §11) to find relevant posts from a global corpus. Each result becomes a `PostCandidate` with `served_type = ForYouPhoenixRetrieval`.

---

## 5. Stage 2 — Candidate Hydration (Feature Enrichment)

**Executed by:** `candidate_pipeline.rs:run_hydrators()` (parallel)

All hydrators run **in parallel**. Each receives the full candidate list and returns an updated list of the same length; the framework merges them via `hydrator.update_all()`.

| Component | File | Data Fetched |
|---|---|---|
| `InNetworkCandidateHydrator` | `candidate_hydrators/in_network_candidate_hydrator.rs` | Marks in-network candidates (`in_network = true`) based on source |
| `CoreDataCandidateHydrator` | `candidate_hydrators/core_data_candidate_hydrator.rs` | Core post data: text, language, media entities — fetched from TweetEntityService (TES) |
| `VideoDurationCandidateHydrator` | `candidate_hydrators/video_duration_candidate_hydrator.rs` | Video duration in milliseconds (used later by `WeightedScorer` for VQV eligibility) |
| `SubscriptionHydrator` | `candidate_hydrators/subscription_hydrator.rs` | Subscription/paywall eligibility flags |
| `GizmoduckCandidateHydrator` | `candidate_hydrators/gizmoduck_hydrator.rs` | Author metadata: username, verification status, account state — from Gizmoduck user service |

**Note:** `VFCandidateHydrator` (visibility filtering data) runs as a *post-selection* hydrator, after the top-K is selected.

---

## 6. Stage 3 — Pre-Scoring Filters

**Executed by:** `candidate_pipeline.rs:run_filters()` (**sequential**)

Filters run one after another. Each receives the candidates that survived all previous filters. On filter failure, the framework falls back to the pre-failure candidate list (error isolation).

| Order | Filter | File | Removes |
|---|---|---|---|
| 1 | `DropDuplicatesFilter` | `filters/drop_duplicates_filter.rs` | Duplicate tweet IDs in the same request |
| 2 | `CoreDataHydrationFilter` | `filters/core_data_hydration_filter.rs` | Candidates for which core data hydration failed |
| 3 | `AgeFilter` | `filters/age_filter.rs` | Posts older than `params::MAX_POST_AGE` |
| 4 | `SelfTweetFilter` | `filters/self_tweet_filter.rs` | Posts authored by the viewer themselves |
| 5 | `RetweetDeduplicationFilter` | `filters/retweet_deduplication_filter.rs` | Multiple reposts pointing to the same original tweet |
| 6 | `IneligibleSubscriptionFilter` | `filters/ineligible_subscription_filter.rs` | Paywalled content that the viewer doesn't have access to |
| 7 | `PreviouslySeenPostsFilter` | `filters/previously_seen_posts_filter.rs` | Posts the viewer has already seen (from `query.seen_ids`) |
| 8 | `PreviouslyServedPostsFilter` | `filters/previously_served_posts_filter.rs` | Posts already served in this session (from `query.served_ids` / bloom filter) |
| 9 | `MutedKeywordFilter` | `filters/muted_keyword_filter.rs` | Posts containing keywords the viewer has muted |
| 10 | `AuthorSocialgraphFilter` | `filters/author_socialgraph_filter.rs` | Posts from blocked or muted authors |

---

## 7. Stage 4 — Scoring and Ranking

**Executed by:** `candidate_pipeline.rs:score()` (**sequential**, scorer order matters)

Scorers run sequentially. Each scorer receives the current candidate list, computes updated scores, and the framework calls `scorer.update_all()` to merge the results.

### Phoenix Scorer (ML Prediction)

**File:** `home-mixer/scorers/phoenix_scorer.rs`

This scorer calls the Phoenix prediction gRPC service (the Grok transformer model, see §11).

**Input to Phoenix:**
```text
user_id + user_action_sequence (engagement history)
candidates: Vec<TweetInfo { tweet_id, author_id }>
```

**Output:** Per-tweet `ActionDistributions` — log-probabilities for each action type.

The scorer converts log-probs to probabilities with `exp()` and populates `PostCandidate::phoenix_scores`:

| Field | Action enum |
|---|---|
| `favorite_score` | `ServerTweetFav` |
| `reply_score` | `ServerTweetReply` |
| `retweet_score` | `ServerTweetRetweet` |
| `photo_expand_score` | `ClientTweetPhotoExpand` |
| `click_score` | `ClientTweetClick` |
| `profile_click_score` | `ClientTweetClickProfile` |
| `vqv_score` | `ClientTweetVideoQualityView` |
| `share_score` | `ClientTweetShare` |
| `share_via_dm_score` | `ClientTweetClickSendViaDirectMessage` |
| `share_via_copy_link_score` | `ClientTweetShareViaCopyLink` |
| `dwell_score` | `ClientTweetRecapDwelled` |
| `quote_score` | `ServerTweetQuote` |
| `quoted_click_score` | `ClientQuotedTweetClick` |
| `follow_author_score` | `ClientTweetFollowAuthor` |
| `not_interested_score` | `ClientTweetNotInterestedIn` |
| `block_author_score` | `ClientTweetBlockAuthor` |
| `mute_author_score` | `ClientTweetMuteAuthor` |
| `report_score` | `ClientTweetReport` |
| `dwell_time` | `DwellTime` (continuous) |

**Retweet handling:** For retweets, the scorer looks up predictions by the *original* tweet ID (`retweeted_tweet_id`) rather than the retweet's own ID.

---

### Weighted Scorer (Score Fusion)

**File:** `home-mixer/scorers/weighted_scorer.rs`

Combines all Phoenix predictions into a single scalar score:

```
weighted_score = Σ (weight_i × P(action_i))
```

Configured weights (defined in `params`):
- **Positive actions** (increase score): `FAVORITE_WEIGHT`, `REPLY_WEIGHT`, `RETWEET_WEIGHT`, `PHOTO_EXPAND_WEIGHT`, `CLICK_WEIGHT`, `PROFILE_CLICK_WEIGHT`, `VQV_WEIGHT`, `SHARE_WEIGHT`, `SHARE_VIA_DM_WEIGHT`, `SHARE_VIA_COPY_LINK_WEIGHT`, `DWELL_WEIGHT`, `QUOTE_WEIGHT`, `QUOTED_CLICK_WEIGHT`, `CONT_DWELL_TIME_WEIGHT`, `FOLLOW_AUTHOR_WEIGHT`
- **Negative actions** (decrease score): `NOT_INTERESTED_WEIGHT`, `BLOCK_AUTHOR_WEIGHT`, `MUTE_AUTHOR_WEIGHT`, `REPORT_WEIGHT`

**VQV eligibility:** The `VQV_WEIGHT` is only applied if `video_duration_ms > params::MIN_VIDEO_DURATION_MS` (prevents rewarding very short videos with the same video-quality-view signal).

**Score normalization:** The raw combined score is passed through `normalize_score()` (from `util::score_normalizer`) and then through `offset_score()`:

```
if weighted_score >= 0:
    final = weighted_score + NEGATIVE_SCORES_OFFSET
else:
    final = (weighted_score + NEGATIVE_WEIGHTS_SUM) / WEIGHTS_SUM * NEGATIVE_SCORES_OFFSET
```

This ensures the final score is always non-negative and that negative-weighted posts are pushed to the bottom without becoming unboundedly negative.

---

### Author Diversity Scorer

**File:** `home-mixer/scorers/author_diversity_scorer.rs`

Prevents the feed from being dominated by a single author. After computing `weighted_score`, this scorer applies a decay multiplier to each candidate's score based on how many other posts from the same author have already appeared higher in the tentative ranking:

```
multiplier(position) = (1 - floor) × decay_factor^position + floor
```

Where:
- `position` = how many posts from the same author precede this one (when sorted by `weighted_score`)
- `decay_factor` = `params::AUTHOR_DIVERSITY_DECAY` (< 1.0, e.g., 0.5)
- `floor` = `params::AUTHOR_DIVERSITY_FLOOR` (minimum multiplier, e.g., 0.1)

For a user's first post: `multiplier = 1.0` (no penalty)
For a user's second post: `multiplier < 1.0` (attenuated)
For a user's third post: even lower multiplier

The final per-candidate field written is `PostCandidate::score`.

---

### OON Scorer (In/Out-of-Network Adjustment)

**File:** `home-mixer/scorers/oon_scorer.rs`

Out-of-network (Phoenix retrieval) candidates have their score multiplied by `params::OON_WEIGHT_FACTOR`:

```rust
match c.in_network {
    Some(false) => base_score * OON_WEIGHT_FACTOR,
    _           => base_score,  // in-network unchanged
}
```

This allows tuning the in-network / out-of-network mix without retraining the ML model.

---

## 8. Stage 5 — Selection

**File:** `home-mixer/selectors/top_k_score_selector.rs`
**Executed by:** `candidate_pipeline.rs:select()`

Sorts all surviving candidates by `PostCandidate::score` (descending) and returns the top `params::RESULT_SIZE` candidates.

---

## 9. Stage 6 — Post-Selection Processing

After the top-K is selected, a second round of hydration and filtering is applied.

**Post-selection hydrators** (parallel):

| Component | File | What It Does |
|---|---|---|
| `VFCandidateHydrator` | `candidate_hydrators/vf_candidate_hydrator.rs` | Calls the Visibility Filtering (VF) service to fetch safety/quality signals for each post |

**Post-selection filters** (sequential):

| Order | Filter | File | Removes |
|---|---|---|---|
| 1 | `VFFilter` | `filters/vf_filter.rs` | Posts flagged by VF as deleted, spam, violence, gore, etc. |
| 2 | `DedupConversationFilter` | `filters/dedup_conversation_filter.rs` | Redundant branches of the same conversation thread (keeps at most one per conversation root) |

After post-selection filters, the final list is truncated to `params::RESULT_SIZE`.

---

## 10. Stage 7 — Side Effects

**File:** `home-mixer/side_effects/cache_request_info_side_effect.rs`
**Executed by:** `candidate_pipeline.rs:run_side_effects()` (async, non-blocking)

Side effects run in a separate Tokio task (`tokio::spawn`) and do **not** block the response.

| Component | What It Does |
|---|---|
| `CacheRequestInfoSideEffect` | Writes request metadata and served tweet IDs to Strato (distributed cache), so that `PreviouslyServedPostsFilter` can exclude them in subsequent requests within the same session |

---

## 11. Phoenix ML Architecture (Deep Dive)

### Ranking Model

**File:** `phoenix/recsys_model.py`
**Class:** `PhoenixModel(hk.Module)`
**Config:** `PhoenixModelConfig`

The ranking model takes a user context and a batch of candidate posts, and outputs per-action engagement logits for each candidate.

**Input data structures:**
- `RecsysBatch` — contains hash indices for user, history posts/authors, and candidate posts/authors, plus multi-hot action vectors and product surface IDs
- `RecsysEmbeddings` — pre-looked-up embedding vectors for all hash indices

**Forward pass (`PhoenixModel.__call__`):**
1. Call `build_inputs()` to construct the full embedding sequence
2. Pass through the transformer (`Transformer.__call__`)
3. Apply layer normalization to the output
4. Slice out candidate positions: `out_embeddings[:, candidate_start_offset:, :]`
5. Apply unembedding projection: `logits = candidate_embeddings @ unembedding_matrix`

**Output:** `RecsysModelOutput(logits)` of shape `[B, num_candidates, num_actions]`

---

### Retrieval Model (Two-Tower)

**File:** `phoenix/recsys_retrieval_model.py`
**Class:** `PhoenixRetrievalModel(hk.Module)`
**Config:** `PhoenixRetrievalModelConfig`

#### User Tower

`build_user_representation()`:
1. Builds user + history embeddings identically to the ranking model (shared `block_user_reduce` and `block_history_reduce` functions)
2. Runs the **same transformer architecture** as the ranker (no `candidate_start_offset`)
3. Performs **masked mean pooling** over all non-padding positions:
   ```python
   user_representation = Σ(output_i × mask_i) / Σ(mask_i)
   ```
4. **L2-normalizes** the result

#### Candidate Tower

`build_candidate_representation()` → `CandidateTower`:
1. Concatenates post and author embeddings: `[post_emb || author_emb]`
2. Applies two-layer MLP with SiLU activation:
   ```
   hidden = silu(post_author_emb @ proj_1)      # [*, emb_size*2]
   output = hidden @ proj_2                      # [*, emb_size]
   ```
3. **L2-normalizes** the result

#### Similarity Search

`_retrieve_top_k()`:
```python
scores = user_representation @ corpus_embeddings.T   # [B, N]
top_k_scores, top_k_indices = jax.lax.top_k(scores, top_k)
```

Both user and candidate representations are L2-normalized, so the dot product is equivalent to cosine similarity. This enables efficient approximate nearest neighbor (ANN) search.

---

### Shared Transformer (Grok Architecture)

**File:** `phoenix/grok.py`
**Class:** `Transformer(hk.Module)`

Architecture details (configurable via `TransformerConfig`):

| Parameter | Description |
|---|---|
| `num_layers` | Number of transformer decoder layers |
| `num_q_heads` | Number of query attention heads |
| `num_kv_heads` | Number of key/value heads (supports grouped-query attention, GQA) |
| `key_size` | Attention key/value dimension |
| `widening_factor` | FFN intermediate size multiplier |
| `attn_output_multiplier` | Scales attention logits before tanh clipping |

**Each `DecoderLayer` (`grok.py:DecoderLayer`) applies:**
1. RMS LayerNorm → Multi-Head Attention (with RoPE) → RMS LayerNorm → residual
2. RMS LayerNorm → SwiGLU FFN → RMS LayerNorm → residual

**Positional encoding:** Rotary Position Embedding (RoPE), implemented in `RotaryEmbedding` (`grok.py`).

**Attention stabilization:** Attention logits are clipped via `tanh`:
```python
attn_logits = 30.0 * tanh(attn_logits / 30.0)
```
This prevents numerical overflow in softmax without a hard clip.

**FFN size** (`ffn_size()`):
```python
_ffn_size = int(widening_factor * emb_size) * 2 // 3
_ffn_size = _ffn_size + (8 - _ffn_size) % 8  # round up to multiple of 8
```

---

### Input Embeddings and Feature Encoding

**File:** `phoenix/recsys_model.py` — `block_user_reduce`, `block_history_reduce`, `block_candidate_reduce`

**Hash-based embeddings:** IDs are not embedded directly; instead, multiple hash functions map each ID to several hash buckets, and the corresponding embeddings are projected into a single `D`-dimensional vector.

| Feature | Encoder | Output shape |
|---|---|---|
| User ID | `block_user_reduce`: concatenate `num_user_hashes` embeddings → linear projection | `[B, 1, D]` |
| History posts | `block_history_reduce`: concatenate post, author, action, surface embeddings → projection | `[B, S, D]` |
| Candidate posts | `block_candidate_reduce`: concatenate post, author, surface embeddings → projection | `[B, C, D]` |

**Action embeddings** (`_get_action_embeddings`):
- Actions are multi-hot vectors (e.g., [1, 0, 1, 0, 0, ...] meaning "liked and retweeted")
- Converted to signed representation: `2 × actions - 1` (maps {0,1} → {-1,+1})
- Projected to `D` dimensions via a learned matrix
- Zeroed out for padding positions (no action taken)

**Product surface embeddings** (`_single_hot_to_embeddings`):
- Categorical index (e.g., Home Timeline = 0, Notifications = 1) → one-hot → embedding lookup

---

### Candidate Isolation Attention Mask

**File:** `phoenix/grok.py` — `make_recsys_attn_mask()`

This is the core algorithmic innovation enabling the transformer to score candidates independently.

**Construction:**
1. Start with a causal (lower-triangular) mask for the full `[user + history + candidates]` sequence
2. Zero out the bottom-right `[C × C]` block (candidate-to-candidate attention)
3. Add back the diagonal of that block (self-attention for each candidate)

```
         Keys: [U | H1 H2 H3 | C1 C2 C3]
Queries:
  U   │  1  │  0  0  0  │  0  0  0
  H1  │  1  │  1  0  0  │  0  0  0
  H2  │  1  │  1  1  0  │  0  0  0
  H3  │  1  │  1  1  1  │  0  0  0
  ----│-----│-----------│----------
  C1  │  1  │  1  1  1  │  1  0  0   ← attends to U+H + self only
  C2  │  1  │  1  1  1  │  0  1  0   ← attends to U+H + self only
  C3  │  1  │  1  1  1  │  0  0  1   ← attends to U+H + self only
```

**Consequence:** The score assigned to candidate Cᵢ is a function **only** of the user + history context plus Cᵢ itself — never of the other candidates in the same batch. This makes scores **consistent** regardless of batch composition, and enables **caching** of scores for recently-seen candidates.

---

## 12. Thunder In-Memory Post Store (Deep Dive)

**Files:** `thunder/main.rs`, `thunder/posts/`, `thunder/thunder_service.rs`, `thunder/kafka/`, `thunder/kafka_utils.rs`

Thunder is the in-network content store. It functions as an always-warm, sub-millisecond read cache for recent posts from followed accounts.

**Initialization sequence** (`thunder/main.rs`):
1. Create `PostStore` with configurable retention period and request timeout
2. Create `StratoClient` for on-demand following-list lookups
3. Start Kafka consumers (configurable thread count via `kafka_num_threads`)
4. Wait for Kafka catch-up signal from **all** consumer threads before declaring readiness
5. Call `post_store.finalize_init()` to atomically flip the store to serving mode
6. Start background `auto_trim` task (runs every 2 minutes to evict expired posts)

**Real-time ingestion:** Kafka topics deliver post-create and post-delete events. Each event is processed and reflected in `PostStore` within milliseconds.

**Query serving** (`thunder_service.rs`):
- Accepts `GetInNetworkPostsRequest { user_id, following_user_ids, max_results, ... }`
- Looks up posts from all followed users in the in-memory store
- Returns the most recent posts up to `max_results`

---

## 13. Candidate Pipeline Framework

**Location:** `candidate-pipeline/`

This is a reusable, generic Rust framework that powers `PhoenixCandidatePipeline`. It separates the *execution semantics* (parallelism, error handling, logging) from the *business logic* (what to fetch, filter, score).

**`CandidatePipeline<Q, C>` trait** (`candidate_pipeline.rs`) defines 9 methods that implementations must provide:

```rust
fn query_hydrators()          -> &[Box<dyn QueryHydrator<Q>>]
fn sources()                  -> &[Box<dyn Source<Q, C>>]
fn hydrators()                -> &[Box<dyn Hydrator<Q, C>>]
fn filters()                  -> &[Box<dyn Filter<Q, C>>]
fn scorers()                  -> &[Box<dyn Scorer<Q, C>>]
fn selector()                 -> &dyn Selector<Q, C>
fn post_selection_hydrators() -> &[Box<dyn Hydrator<Q, C>>]
fn post_selection_filters()   -> &[Box<dyn Filter<Q, C>>]
fn side_effects()             -> Arc<Vec<Box<dyn SideEffect<Q, C>>>>
```

**Execution semantics enforced by the framework:**

| Stage | Execution Model | Failure Behavior |
|---|---|---|
| Query Hydration | Parallel (`join_all`) | Log error, skip failed hydrator |
| Source Fetching | Parallel (`join_all`) | Log error, exclude source's candidates |
| Candidate Hydration | Parallel (`join_all`) | Log error, skip failed hydrator; length mismatch → skip |
| Filtering | Sequential | Log error, restore pre-failure candidate list |
| Scoring | Sequential | Log error, skip failed scorer; length mismatch → skip |
| Selection | Single call | N/A |
| Post-sel Hydration | Parallel (`join_all`) | Same as candidate hydration |
| Post-sel Filtering | Sequential | Same as filtering |
| Side Effects | Async (spawned task) | Non-blocking; errors silently dropped |

Each component interface has an `enable(&query) -> bool` method (default: `true`) allowing components to opt out for certain request types without being removed from the pipeline.

---

## 14. Complete Data Flow Summary

```
 gRPC: GetScoredPosts(viewer_id, seen_ids, served_ids, ...)
                          │
              ┌───────────▼────────────┐
              │  QUERY HYDRATION        │  ← parallel
              │  UserActionSeqHydrator  │    engagement history
              │  UserFeaturesHydrator   │    following list, muted kws
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  CANDIDATE SOURCING     │  ← parallel
              │  ThunderSource          │    ~500 in-network posts
              │  PhoenixSource          │    ~1500 out-of-network posts
              └───────────┬────────────┘
                          │ ~2000 raw candidates
              ┌───────────▼────────────┐
              │  CANDIDATE HYDRATION    │  ← parallel
              │  InNetworkHydrator      │    in_network flag
              │  CoreDataHydrator       │    text, media, language
              │  VideoDurationHydrator  │    video_duration_ms
              │  SubscriptionHydrator   │    paywall flags
              │  GizmoduckHydrator      │    author username, verified
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  PRE-SCORING FILTERS    │  ← sequential
              │  [10 filters]           │    dedup, age, self, etc.
              └───────────┬────────────┘
                          │ ~800–1500 filtered candidates
              ┌───────────▼────────────┐
              │  SCORING                │  ← sequential
              │  1. PhoenixScorer       │    ML: P(like), P(reply), ...
              │  2. WeightedScorer      │    weighted sum of probabilities
              │  3. AuthorDiversityScorer│   decay repeated authors
              │  4. OONScorer           │    adjust out-of-network scores
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  SELECTION              │
              │  TopKScoreSelector      │    top RESULT_SIZE by score
              └───────────┬────────────┘
                          │ top-K candidates
              ┌───────────▼────────────┐
              │  POST-SELECTION         │  ← parallel hydration
              │  VFCandidateHydrator    │    safety/quality signals
              │  VFFilter               │    remove flagged content
              │  DedupConversationFilter│    remove duplicate threads
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  SIDE EFFECTS           │  ← async (non-blocking)
              │  CacheRequestInfoSideEffect │ store served IDs
              └────────────────────────┘
                          │
              gRPC response: Vec<ScoredPost>
```

---

## 15. Key Engineering Decisions

### 1. No Hand-Engineered Ranking Features
The Phoenix transformer learns to rank entirely from the raw engagement history sequence. There are no manually crafted content features (e.g., recency boosts, topic matching). The only hand-tuned component is the weight vector in `WeightedScorer`.

### 2. Candidate Isolation via Custom Attention Mask
`make_recsys_attn_mask()` (`grok.py`) prevents cross-candidate attention. This means scores are batch-agnostic: the score for post A is the same regardless of which other posts are in the batch. This property enables **score caching** and makes online A/B testing significantly easier.

### 3. Shared Architecture Across Retrieval and Ranking
The user tower in `PhoenixRetrievalModel` uses the identical `Transformer` class as `PhoenixModel`. The only difference is: retrieval uses mean-pooled output + L2 normalization for ANN search, while ranking uses per-position outputs for multi-action prediction.

### 4. Hash-Based Embeddings (No ID Lookup Tables)
IDs are mapped to embedding vectors using multiple hash functions rather than a single large lookup table. This avoids cold-start problems for new posts/users and bounds memory usage by the hash table size, not the ID space.

### 5. Two-Stage Pipeline (Retrieval → Ranking)
- **Retrieval** (Phoenix two-tower + Thunder) handles millions of candidates efficiently using approximate nearest neighbor search
- **Ranking** (Phoenix transformer) applies the more expensive model to only the hundreds of candidates that survived retrieval and filtering

### 6. Parallel Sources, Sequential Filters and Scorers
Sources and hydrators run in parallel to minimize latency. Filters and scorers run sequentially because each stage depends on the output of the previous one (filters reduce the candidate set; scorers accumulate fields).

### 7. Diversity Enforcement Post-Scoring
Author diversity is applied *after* ML scoring, not as a hard filter. This preserves ML score signal while still preventing feed monopolization, and allows tuning the diversity/relevance trade-off via `AUTHOR_DIVERSITY_DECAY` and `AUTHOR_DIVERSITY_FLOOR` parameters.

### 8. Graceful Degradation
The framework is designed to degrade gracefully: if a hydrator or scorer fails, its results are skipped and the pipeline continues with the pre-failure state. This ensures a response is always returned, even if some features are unavailable.

### 9. Out-of-Network Content Gating
`PhoenixSource.enable()` returns `false` when `in_network_only = true`. This allows clients to request a pure in-network feed (e.g., the "Following" tab) using the same infrastructure with no code changes.

### 10. Safety Filtering Deferred to Post-Selection
Visibility Filtering (`VFCandidateHydrator` + `VFFilter`) runs *after* the top-K is selected. This avoids the latency cost of calling the VF service for all candidates, while ensuring all served posts pass safety checks.
