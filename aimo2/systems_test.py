from vllm import CompletionOutput, RequestOutput

from .systems import RequestInput


def test_request_input():
    request_input = RequestInput()
    request_input = request_input.add_system_message("You are a helpful assistant.")

    request_input = request_input.add_user_message("Hello, how are you?")
    request_input = request_input.add_request_output(
        RequestOutput(
            request_id="1",
            prompt="Hello, how are you?",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="Good and you?",
                    token_ids=[1, 2, 3],
                    cumulative_logprob=0.1,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )
    )
    request_input = request_input.with_seed(123)
    assert request_input.conversation == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "Good and you?"},
    ]
    assert request_input.sampling_params.seed == 123
