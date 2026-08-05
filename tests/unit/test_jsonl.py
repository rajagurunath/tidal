"""Tests for batch-input JSONL validation and prefix grouping (Task A2)."""

from __future__ import annotations

import json
import time

import pytest

from tidal.api.jsonl import BatchInputError, ParsedLine, parse_batch_input

CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"


def chat_line(custom_id: str, content: str = "hello", **body_extra: object) -> str:
    body = {"model": "m", "messages": [{"role": "user", "content": content}]}
    body.update(body_extra)
    return json.dumps({"custom_id": custom_id, "method": "POST", "url": CHAT, "body": body})


def completion_line(custom_id: str, prompt: str = "hello", **body_extra: object) -> str:
    body = {"model": "m", "prompt": prompt}
    body.update(body_extra)
    return json.dumps(
        {"custom_id": custom_id, "method": "POST", "url": COMPLETIONS, "body": body}
    )


def blob(*lines: str) -> bytes:
    return ("\n".join(lines) + "\n").encode()


# -- happy path ------------------------------------------------------------


def test_valid_chat_file_parses_every_line():
    parsed = parse_batch_input(blob(chat_line("a"), chat_line("b", "other")), CHAT)
    assert [p.custom_id for p in parsed] == ["a", "b"]
    assert [p.line_no for p in parsed] == [1, 2]
    assert parsed[0].body["model"] == "m"


def test_parsed_line_unpacks_as_repository_item_tuple():
    (line,) = parse_batch_input(blob(chat_line("a")), CHAT)
    assert isinstance(line, ParsedLine)
    custom_id, line_no, prefix_group, body = tuple(line)
    assert (custom_id, line_no, prefix_group) == ("a", 1, 0)
    assert isinstance(body, dict)


def test_blank_lines_are_skipped_but_line_numbers_stay_physical():
    raw = ("\n" + chat_line("a") + "\n\n   \n" + chat_line("b") + "\n").encode()
    parsed = parse_batch_input(raw, CHAT)
    assert [p.line_no for p in parsed] == [2, 5]


def test_completions_endpoint_accepts_prompt_bodies():
    parsed = parse_batch_input(blob(completion_line("a")), COMPLETIONS)
    assert parsed[0].body["prompt"] == "hello"


def test_n_equal_to_one_is_allowed():
    parsed = parse_batch_input(blob(chat_line("a", n=1)), CHAT)
    assert parsed[0].body["n"] == 1


def test_structured_content_parts_are_accepted():
    content = [{"type": "text", "text": "hi"}]
    raw = json.dumps(
        {
            "custom_id": "a",
            "method": "POST",
            "url": CHAT,
            "body": {"model": "m", "messages": [{"role": "user", "content": content}]},
        }
    )
    parsed = parse_batch_input(blob(raw), CHAT)
    assert parsed[0].prefix_group == 0


# -- rejections ------------------------------------------------------------


def test_empty_file_is_rejected():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(b"\n  \n", CHAT)
    assert exc.value.line_no == 0


def test_bad_json_reports_its_line_number():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(chat_line("a"), "{not json", chat_line("c")), CHAT)
    assert exc.value.line_no == 2
    assert "json" in exc.value.reason.lower()


def test_non_object_line_is_rejected():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob("[1, 2, 3]"), CHAT)
    assert exc.value.line_no == 1


def test_missing_custom_id_is_rejected():
    raw = json.dumps({"method": "POST", "url": CHAT, "body": {"model": "m", "messages": []}})
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(raw), CHAT)
    assert exc.value.line_no == 1
    assert "custom_id" in exc.value.reason


def test_duplicate_custom_id_reports_the_second_line():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(chat_line("a"), chat_line("b"), chat_line("a")), CHAT)
    assert exc.value.line_no == 3
    assert "duplicate" in exc.value.reason.lower()


def test_non_post_method_is_rejected():
    raw = json.dumps(
        {
            "custom_id": "a",
            "method": "GET",
            "url": CHAT,
            "body": {"model": "m", "messages": []},
        }
    )
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(raw), CHAT)
    assert exc.value.line_no == 1
    assert "method" in exc.value.reason.lower()


def test_url_must_match_the_batch_endpoint():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(chat_line("a"), completion_line("b")), CHAT)
    assert exc.value.line_no == 2
    assert "url" in exc.value.reason.lower()


def test_streaming_requests_are_rejected():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(chat_line("a", stream=True)), CHAT)
    assert exc.value.line_no == 1
    assert "stream" in exc.value.reason.lower()


def test_n_greater_than_one_is_rejected():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(chat_line("a"), chat_line("b", n=2)), CHAT)
    assert exc.value.line_no == 2
    assert exc.value.reason.startswith("n")


def test_missing_model_is_rejected():
    raw = json.dumps(
        {"custom_id": "a", "method": "POST", "url": CHAT, "body": {"messages": []}}
    )
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(raw), CHAT)
    assert "model" in exc.value.reason


def test_chat_body_without_messages_is_rejected():
    raw = json.dumps(
        {"custom_id": "a", "method": "POST", "url": CHAT, "body": {"model": "m"}}
    )
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(raw), CHAT)
    assert "messages" in exc.value.reason


def test_completions_body_without_prompt_is_rejected():
    raw = json.dumps(
        {"custom_id": "a", "method": "POST", "url": COMPLETIONS, "body": {"model": "m"}}
    )
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(raw), COMPLETIONS)
    assert "prompt" in exc.value.reason


def test_missing_body_is_rejected():
    raw = json.dumps({"custom_id": "a", "method": "POST", "url": CHAT})
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(blob(raw), CHAT)
    assert "body" in exc.value.reason


def test_unsupported_endpoint_is_rejected():
    with pytest.raises(BatchInputError):
        parse_batch_input(blob(chat_line("a")), "/v1/embeddings")


def test_invalid_utf8_is_rejected():
    with pytest.raises(BatchInputError) as exc:
        parse_batch_input(b"\xff\xfe not utf8", CHAT)
    assert exc.value.line_no == 0


# -- prefix grouping -------------------------------------------------------


def test_shared_prefix_lines_share_a_group_in_first_seen_order():
    parsed = parse_batch_input(
        blob(
            chat_line("a", "system prompt A"),
            chat_line("b", "system prompt B"),
            chat_line("c", "system prompt A"),
            chat_line("d", "system prompt C"),
        ),
        CHAT,
    )
    assert [p.prefix_group for p in parsed] == [0, 1, 0, 2]


def test_grouping_only_considers_the_first_256_characters():
    shared = "x" * 256
    parsed = parse_batch_input(
        blob(chat_line("a", shared + "tail-one"), chat_line("b", shared + "tail-two")),
        CHAT,
    )
    assert [p.prefix_group for p in parsed] == [0, 0]


def test_grouping_splits_when_the_first_256_characters_differ():
    parsed = parse_batch_input(
        blob(chat_line("a", "y" + "x" * 300), chat_line("b", "z" + "x" * 300)), CHAT
    )
    assert [p.prefix_group for p in parsed] == [0, 1]


def test_grouping_uses_only_the_first_message():
    a = json.dumps(
        {
            "custom_id": "a",
            "method": "POST",
            "url": CHAT,
            "body": {
                "model": "m",
                "messages": [{"role": "system", "content": "shared"},
                             {"role": "user", "content": "q1"}],
            },
        }
    )
    b = json.dumps(
        {
            "custom_id": "b",
            "method": "POST",
            "url": CHAT,
            "body": {
                "model": "m",
                "messages": [{"role": "system", "content": "shared"},
                             {"role": "user", "content": "q2"}],
            },
        }
    )
    parsed = parse_batch_input(blob(a, b), CHAT)
    assert [p.prefix_group for p in parsed] == [0, 0]


def test_completions_grouping_uses_the_prompt_string():
    parsed = parse_batch_input(
        blob(
            completion_line("a", "shared prompt"),
            completion_line("b", "other prompt"),
            completion_line("c", "shared prompt"),
        ),
        COMPLETIONS,
    )
    assert [p.prefix_group for p in parsed] == [0, 1, 0]


def test_empty_messages_list_groups_together():
    raw = json.dumps(
        {"custom_id": "a", "method": "POST", "url": CHAT, "body": {"model": "m", "messages": []}}
    )
    parsed = parse_batch_input(blob(raw), CHAT)
    assert parsed[0].prefix_group == 0


# -- performance -----------------------------------------------------------


def test_ten_thousand_lines_parse_in_under_a_second():
    lines = [chat_line(f"id-{i}", f"prompt {i % 50} " + "z" * 200) for i in range(10_000)]
    raw = blob(*lines)
    start = time.perf_counter()
    parsed = parse_batch_input(raw, CHAT)
    elapsed = time.perf_counter() - start
    assert len(parsed) == 10_000
    assert len({p.prefix_group for p in parsed}) == 50
    assert elapsed < 1.0, f"parse took {elapsed:.3f}s"
