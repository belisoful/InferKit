# The Swift-only surface

What Foundation Models offers that no request key can carry, what it is for, and what this package
does instead.

## Overview

Every option the backend supports is a core request key, so an Objective-C app configures Apple's
model the way it configures any other engine. Part of the framework cannot become a key: a result
builder is a closure over types, a macro runs at compile time, and a generic needs a type the caller
names. Those stay Swift, and this article says what each one does and where the contract's path
runs instead.

The list is closed. Anything else the framework offers has a key, and a new key is the right fix for
anything missing.

### Dynamic instructions and profiles

`DynamicInstructions` is a result-builder protocol: a type with a `body` built from
`@DynamicInstructionsBuilder`, so instructions are assembled from tools and conditions at run time
rather than written as one string. `LanguageModelSession.Profile` and the `DynamicProfile` protocol
wrap the same idea with sampling and tools attached, and
`LanguageModelSession(model:dynamicInstructions:history:)` opens a session on one.

A result builder has no dictionary form, so there is nothing to put in a request. The contract's path
is a system message: the first `system` entry in `NFKInputMessages` becomes the session's
instructions, `NFKParameterTools` declares the tools for that request, and the sampling keys carry
what a profile would have set. A caller that assembles instructions from several pieces joins them
before the request, which is what the builder does anyway.

### `@Generable(name:)`

The `@Generable` macro makes a Swift type the model can fill, and macOS 27 adds a `name:` argument
that fixes the schema's name rather than taking the type's. It is a macro on a declaration, so it
needs a compile-time type.

The contract's path is `NFKParameterJSONSchema`, which names the schema in the JSON itself and
returns the parsed object under `NFKOutputStructured`. The runtime schema covers what a dynamic
caller needs; `@Generable` covers what a Swift caller wants at compile time. See
<doc:ToolsAndStructuredOutput>.

### `ImageReference`

A `Generable` whose `attachmentLabel` lets the model point back at an image it was given, so a reply
can say which picture it means. It is a generable type, so a caller needs the Swift type to read one
out of a response.

Images reach the model through `NFKInputImage` and `NFKInputImages` on macOS 27, and a reply that
must name one is the case that needs Swift.

### Session properties

A `LanguageModelSession` carries `isResponding`, its `transcript`, `usage` (macOS 27), and
`prewarm(promptPrefix:)`. The backend owns a session per request, so none of them is a property of
anything a request can name.

What each becomes: streaming arrives through the job's `partialResult` rather than `isResponding`,
the transcript is the `NFKInputMessages` the caller already holds, `usage` comes back under
`NFKOutputUsage`, and `prepare()` prewarms the model once.

### `transcriptErrorHandlingPolicy`

On macOS 27 a session chooses `.revertTranscript` or `.preserveTranscript`, deciding whether a failed
turn stays in the session's history. The backend builds a transcript per request and discards it, so
there is no history for a policy to govern.

The contract's path is the same mechanism from the other end: the caller owns the conversation and
sends it with each request, so after a failure it decides what to resend. That is
`.revertTranscript` by default and `.preserveTranscript` by keeping the failed turn in the array.

## Reaching them anyway

A Swift app that wants any of this uses `LanguageModelSession` directly, and it loses nothing by
doing so: ``NFKInferKitLanguageModel`` puts an InferKit backend behind that same session, so one app
can drive Apple's model through the framework's Swift API and a remote endpoint or an MLX model
through the same code. See <doc:ProviderBridge>.

## Topics

### Related

- <doc:ToolsAndStructuredOutput>
- <doc:ProviderBridge>
