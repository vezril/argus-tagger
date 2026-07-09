# suggestion-contract

The messages Argus consumes and produces, exchanged over HermesMQ with Artemis.

> **Envelope encoding.** Hermes treats a message body as an opaque UTF-8 string `payload`
> plus a `Map<string,string> attributes` — it knows nothing of `TagJob`/`TagSuggestions`. So
> the structured message SHALL be **JSON-encoded into the `payload` string**, and `postId`
> SHALL be **duplicated into `attributes`** (cheap correlation/filtering without decoding the
> body). Both Argus and Artemis SHALL encode/decode against this same convention.

## ADDED Requirements

### Requirement: Tag-job message consumed from Artemis

Argus SHALL consume a `TagJob` message from the `media.tag` topic carrying at least the
`postId`, the Apollo reference to the **sample** derivative (`bucket` + `object`), and the
`mediaType`. The `postId` SHALL serve as the idempotency key.

#### Scenario: A tag-job references the sample to tag
- **GIVEN** a post whose sample derivative is stored in Apollo
- **WHEN** Artemis publishes its `TagJob`
- **THEN** the message carries the `postId` and the Apollo `bucket`/`object` of the sample, which Argus fetches to tag

#### Scenario: Edge case — an unreadable sample yields a failed result, not a wedge
- **GIVEN** a `TagJob` whose sample cannot be fetched or decoded
- **WHEN** Argus processes it
- **THEN** it publishes a `TagSuggestions` with `status: failed` (or nacks for retry) and continues the queue

### Requirement: Tag-suggestions message published to Artemis

Argus SHALL publish a `TagSuggestions` message carrying the `postId`, a list of raw
suggestions each with `{tag, category?, confidence, source}` (`source` ∈ `wd` | `ram`), an
optional `rating`, and a `status` (`ok` | `failed`). Suggestions SHALL be **raw** — Argus
normalizes surface form only; meaning-level namespace merging is Artemis's job.

#### Scenario: Suggestions carry confidence and source model
- **GIVEN** a tagged image
- **WHEN** Argus publishes `TagSuggestions`
- **THEN** each suggestion includes its confidence and which model produced it (`wd` or `ram`)

#### Scenario: Edge case — a tag seen by both models appears from both sources
- **GIVEN** both wd-tagger and RAM++ emit the same normalized tag
- **WHEN** Argus builds the suggestion list
- **THEN** it may include both source entries (or one merged entry keeping the max confidence); Artemis makes the final canonical merge
