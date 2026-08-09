from dataclasses import dataclass

from httk.data.store_common import SaveProjection


@dataclass(frozen=True)
class _SavedRecord:
    value: int


def test_save_projection_uses_the_trusted_content_id_route() -> None:
    source = _SavedRecord(17)
    first = SaveProjection().content_id(_SavedRecord, source)
    second = SaveProjection().content_id(_SavedRecord, source)

    assert first == second
    assert source._httk_cached_content_ids[1][_SavedRecord] == first
