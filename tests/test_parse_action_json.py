"""Unit tests for _parse_action_json and _normalize_response."""

from agents.base import _normalize_response, _parse_action_json

# ── _parse_action_json ────────────────────────────────────────────────────────


class TestParseActionJson:
    def test_clean_finish(self):
        text = '{"thought": "done", "action": "finish", "answer": "42", "confidence": 0.9}'
        result = _parse_action_json(text)
        assert result == {"thought": "done", "action": "finish", "answer": "42", "confidence": 0.9}

    def test_clean_tool_call(self):
        text = '{"thought": "need data", "action": "http_fetch", "args": {"url": "https://example.com"}}'
        result = _parse_action_json(text)
        assert result["action"] == "http_fetch"
        assert result["args"]["url"] == "https://example.com"

    def test_parallel_actions(self):
        text = '{"thought": "fan out", "actions": [{"tool": "shell", "args": {"cmd": "ls"}}, {"tool": "http_fetch", "args": {"url": "https://x.com"}}]}'
        result = _parse_action_json(text)
        assert len(result["actions"]) == 2

    def test_extra_tokens_after_json(self):
        """gpt-5.5 streams a separator + second object after the valid action."""
        text = (
            '{"thought": "fetch it", "action": "http_fetch", "args": {"url": "https://x.com"}}'
            "\n?\n"
            '{"thought": "I cannot complete the request because the context window is full."}'
        )
        result = _parse_action_json(text)
        assert result is not None
        assert result["action"] == "http_fetch"

    def test_malformed_preamble_then_valid_action(self):
        """Model emits a truncated/invalid JSON block before the real action object."""
        text = (
            '{"thought": "Need to gather info.\n'  # unescaped newline → invalid JSON
            '{"action": "shell", "args": {"cmd": "uname -a"}, "thought": "t"}'
        )
        result = _parse_action_json(text)
        assert result is not None
        assert result["action"] == "shell"

    def test_extra_whitespace_and_prose_before_json(self):
        text = '  Here is my response:\n{"action": "finish", "answer": "hi", "thought": "t"}'
        result = _parse_action_json(text)
        assert result["action"] == "finish"

    def test_nested_object_in_args(self):
        text = '{"action": "tool", "args": {"nested": {"a": 1, "b": [2, 3]}}}'
        result = _parse_action_json(text)
        assert result["args"]["nested"]["b"] == [2, 3]

    def test_brace_inside_string_value(self):
        text = '{"action": "finish", "answer": "result is {42}", "thought": "t"}'
        result = _parse_action_json(text)
        assert result["answer"] == "result is {42}"

    def test_escaped_quote_inside_string(self):
        text = r'{"action": "finish", "answer": "she said \"hello\"", "thought": "t"}'
        result = _parse_action_json(text)
        assert result["answer"] == 'she said "hello"'

    def test_empty_string_returns_none(self):
        assert _parse_action_json("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_action_json("   \n\t  ") is None

    def test_plain_text_no_json_returns_none(self):
        assert _parse_action_json("sorry, I don't know") is None

    def test_array_at_root_returns_none(self):
        assert _parse_action_json("[1, 2, 3]") is None

    def test_truncated_json_returns_none(self):
        assert _parse_action_json('{"action": "finish"') is None

    def test_multiple_top_level_objects_picks_first(self):
        text = '{"action": "a"} {"action": "b"}'
        result = _parse_action_json(text)
        assert result["action"] == "a"

    def test_unicode_in_values(self):
        text = '{"action": "finish", "answer": "こんにちは", "thought": "t"}'
        result = _parse_action_json(text)
        assert result["answer"] == "こんにちは"


# ── _normalize_response ───────────────────────────────────────────────────────


class TestNormalizeResponse:
    def test_passthrough_action_dict(self):
        d = {"action": "finish", "answer": "x", "thought": "t"}
        assert _normalize_response(d) is d

    def test_passthrough_actions_dict(self):
        d = {"actions": [{"tool": "shell", "args": {}}], "thought": "t"}
        assert _normalize_response(d) is d

    def test_dict_with_text_key(self):
        d = {"text": '{"action": "finish", "answer": "y", "thought": "t"}'}
        result = _normalize_response(d)
        assert result["action"] == "finish"

    def test_string_response(self):
        s = '{"action": "finish", "answer": "z", "thought": "t"}'
        result = _normalize_response(s)
        assert result["action"] == "finish"

    def test_string_with_extra_tokens(self):
        s = '{"action": "http_fetch", "args": {"url": "u"}, "thought": "t"}\n?\n{"thought": "oops"}'
        result = _normalize_response(s)
        assert result["action"] == "http_fetch"

    def test_invalid_returns_none(self):
        assert _normalize_response("not json at all") is None

    def test_none_input_returns_none(self):
        assert _normalize_response(None) is None
