import json

from argus.contract import (
    POST_ID_ATTRIBUTE,
    SampleRef,
    Source,
    Status,
    Suggestion,
    TagJob,
    TagSuggestions,
)


def test_tagjob_decodes_from_payload() -> None:
    payload = json.dumps(
        {
            "postId": "post-42",
            "sample": {"bucket": "samples", "object": "post-42/sample.webp"},
            "mediaType": "image/webp",
        }
    )
    job = TagJob.from_payload(payload)
    assert job == TagJob(
        post_id="post-42",
        sample=SampleRef(bucket="samples", object="post-42/sample.webp"),
        media_type="image/webp",
    )


def test_tagsuggestions_roundtrips_through_payload() -> None:
    suggestions = TagSuggestions(
        post_id="post-42",
        suggestions=[
            Suggestion(tag="tree", confidence=0.912345, source=Source.RAM),
            Suggestion(tag="1girl", confidence=0.98, source=Source.WD, category="general"),
        ],
    )
    body = json.loads(suggestions.to_payload())
    assert body["postId"] == "post-42"
    assert body["status"] == "ok"
    assert body["suggestions"][0] == {"tag": "tree", "confidence": 0.9123, "source": "ram"}
    assert body["suggestions"][1]["category"] == "general"


def test_post_id_mirrored_into_attributes() -> None:
    suggestions = TagSuggestions(post_id="post-42", suggestions=[])
    assert suggestions.attributes() == {POST_ID_ATTRIBUTE: "post-42"}


def test_failed_result_is_empty_and_flagged() -> None:
    failed = TagSuggestions.failed("post-99")
    assert failed.status is Status.FAILED
    assert failed.suggestions == []
    assert json.loads(failed.to_payload())["status"] == "failed"
