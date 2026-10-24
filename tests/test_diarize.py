from clinical_scribe.diarize import _Block, assign_roles


def _block(cluster: int, text: str, start: float = 0.0) -> _Block:
    return _Block(start=start, end=start + 1, cluster=cluster, text=text)


def test_assign_roles_uses_question_density():
    blocks = [
        _block(0, "What brings you in?", 0),
        _block(1, "I have a headache.", 1),
        _block(0, "How long? Any vomiting?", 2),
        _block(1, "Three days. No vomiting.", 3),
    ]
    assert assign_roles(blocks) == {0: "clinician", 1: "patient"}


def test_assign_roles_tie_falls_back_to_first_speaker():
    blocks = [
        _block(1, "Hello.", 0),
        _block(0, "Hi.", 1),
        _block(1, "I have pain.", 2),
        _block(0, "Okay.", 3),
    ]
    assert assign_roles(blocks) == {1: "clinician", 0: "patient"}
