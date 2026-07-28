from agent.graph import _stream_delta_piece


def test_stream_delta_piece_accepts_normal_increment():
    assert _stream_delta_piece("", "第一段") == "第一段"
    assert _stream_delta_piece("第一段", "第二段") == "第二段"


def test_stream_delta_piece_trims_cumulative_content():
    previous = "计划：先 getTrack，再 getFrames"
    incoming = "计划：先 getTrack，再 getFrames，然后 dedupTracks"
    assert _stream_delta_piece(previous, incoming) == "，然后 dedupTracks"


def test_stream_delta_piece_trims_overlap_content():
    previous = "先 getTrack，再 getFrames"
    incoming = "getFrames，然后 dedupTracks"
    assert _stream_delta_piece(previous, incoming) == "，然后 dedupTracks"


def test_stream_delta_piece_drops_exact_repeat():
    previous = "然后 getFrames，然后 dedupTracks"
    assert _stream_delta_piece(previous, "getFrames，然后 dedupTracks") == ""
    assert _stream_delta_piece(previous, previous) == ""


def test_stream_delta_piece_handles_previous_inside_new_cumulative_window():
    previous = "再 getFrames，然后 dedupTracks"
    incoming = "计划：先 getTrack，再 getFrames，然后 dedupTracks，最后统计数量"
    assert _stream_delta_piece(previous, incoming) == "，最后统计数量"
