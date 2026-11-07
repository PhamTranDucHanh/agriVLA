from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import pytest

from agri_vla.preflight import project_root

ROOT = project_root()
SCHEMA = json.loads((ROOT / "schemas/scene_v0.schema.json").read_text(encoding="utf-8"))
EXAMPLES = [json.loads(line) for line in (ROOT / "examples/scene_pair.example.jsonl").read_text(encoding="utf-8").splitlines()]


def test_schema_and_bilingual_examples_are_valid():
    Draft202012Validator.check_schema(SCHEMA)
    validator = Draft202012Validator(SCHEMA)
    for record in EXAMPLES:
        validator.validate(record)
    vi, en = EXAMPLES
    assert {vi["language"], en["language"]} == {"vi", "en"}
    for key in ("scene_id", "images", "source_image_ids", "intent", "intent_params", "target_action", "translation_pair_id"):
        assert vi[key] == en[key]
    assert vi["sample_id"] != en["sample_id"]


@pytest.mark.parametrize("path", ["/tmp/image.png", "../raw/image.png", "raw/../image.png", "C:/image.png", "raw\\image.png"])
def test_schema_rejects_nonportable_image_paths(path):
    record = deepcopy(EXAMPLES[0])
    record["images"]["zone1"] = path
    assert not Draft202012Validator(SCHEMA).is_valid(record)


@pytest.mark.parametrize("key,value", [
    ("target_action", [0, 1, 4]),
    ("target_action", [0, 1]),
    ("language", "fr"),
    ("instruction", "   "),
    ("intent_params", {"zone": 4}),
    ("intent_params", {}),
    ("intent_params", {"zone": 2, "cap": 1}),
    ("source_image_ids", ["same", "same", "same"]),
    ("wording_partition", "unseen"),
    ("teacher_annotations", {"wilt_scores": [0, 2, 5]}),
])
def test_schema_rejects_invalid_contract_values(key, value):
    record = deepcopy(EXAMPLES[0])
    record[key] = value
    assert not Draft202012Validator(SCHEMA).is_valid(record)
