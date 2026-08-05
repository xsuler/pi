import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { Context, Model } from "../src/types.ts";

const mockState = vi.hoisted(() => ({
	completion: {} as Record<string, unknown>,
	request: undefined as Record<string, unknown> | undefined,
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (request: Record<string, unknown>) => {
					mockState.request = request;
					const promise = Promise.resolve(mockState.completion) as Promise<Record<string, unknown>> & {
						withResponse: () => Promise<{
							data: Record<string, unknown>;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: mockState.completion,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}
	return { default: FakeOpenAI };
});

const model: Model<"openai-completions"> = {
	id: "policy",
	name: "Policy",
	api: "openai-completions",
	provider: "areno",
	baseUrl: "http://127.0.0.1:8000/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 4096,
	compat: { supportsStreaming: false },
};

const context: Context = {
	messages: [{ role: "user", content: "inspect the repository", timestamp: 1 }],
};

describe("OpenAI completions non-streaming compatibility", () => {
	beforeEach(() => {
		mockState.request = undefined;
		mockState.completion = {
			id: "chatcmpl-areno",
			model: "policy",
			choices: [
				{
					index: 0,
					message: {
						role: "assistant",
						content: null,
						tool_calls: [
							{
								id: "call-1",
								type: "function",
								function: { name: "read", arguments: '{"path":"README.md"}' },
							},
						],
					},
					finish_reason: "tool_calls",
				},
			],
			usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 },
			areno: { input_tokens: [1, 2], response_tokens: [3, 4], response_logprobs: [-0.1, -0.2] },
		};
	});

	it("requests a non-streaming completion and preserves training metadata", async () => {
		const message = await streamOpenAICompletions(model, context, { apiKey: "areno-agentic" }).result();

		expect(mockState.request?.stream).toBe(false);
		expect(mockState.request?.stream_options).toBeUndefined();
		expect(message.stopReason).toBe("toolUse");
		expect(message.content).toEqual([
			{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "README.md" } },
		]);
		expect(message.providerMetadata).toEqual({
			areno: { input_tokens: [1, 2], response_tokens: [3, 4], response_logprobs: [-0.1, -0.2] },
		});
	});
});
