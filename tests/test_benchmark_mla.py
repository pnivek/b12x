from __future__ import annotations

from benchmarks import benchmark_mla


def test_build_decode_cases_covers_requested_matrix_and_topk_cap() -> None:
    cases = benchmark_mla._build_decode_cases(
        modes=["decode"],
        batch_sizes=[1, 2, 4, 8],
        cache_lens=[1024, 32768, 131072],
        verify_q_lens=[benchmark_mla.DEFAULT_PREFILL_Q_LENS[0]],
        topk_cap=2048,
    )

    assert len(cases) == 12
    assert cases[0] == benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024)
    assert cases[1] == benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=32768, topk=2048)
    assert cases[-1] == benchmark_mla.DecodeCase(mode="decode", batch_size=8, cache_len=131072, topk=2048)


def test_build_decode_cases_skips_prefill_chunks_larger_than_context() -> None:
    cases = benchmark_mla._build_decode_cases(
        modes=["decode", "prefill"],
        batch_sizes=[1],
        cache_lens=[1024, 32768],
        verify_q_lens=[16384],
        topk_cap=2048,
    )

    assert cases == [
        benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024),
        benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=32768, topk=2048),
        benchmark_mla.DecodeCase(
            mode="prefill",
            batch_size=1,
            cache_len=32768,
            topk=2048,
            q_len=16384,
        ),
    ]


def test_render_case_line_reports_public_step_metrics() -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(mode="prefill", batch_size=4, cache_len=32768, topk=2048, q_len=16384),
        graph_width=32768,
        metadata_us=12.34,
        replay_us=567.89,
        indexer_us=123.45,
        mla_us=456.78,
        split_enabled=True,
        chunk_size=256,
        num_chunks=8,
        mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
    )

    line = benchmark_mla._render_case_line(report)

    assert "glm51-prefill tp8" in line
    assert "bs= 4" in line
    assert "q=16384" in line
    assert "ctx= 32768" in line
    assert "graphw= 32768" in line
    assert "topk=2048" in line
    assert "split= on" in line
    assert "chunk=256" in line
    assert "nchunks=8" in line
    assert "step=" in line
    assert "meta=" in line
    assert "replay=" in line
    assert "indexer=" in line
    assert "mla=" in line


def test_render_summary_lines_reports_geomeans() -> None:
    reports = [
        benchmark_mla.CaseReport(
            case=benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024),
            graph_width=8192,
            metadata_us=100.0,
            replay_us=400.0,
            indexer_us=100.0,
            mla_us=400.0,
            split_enabled=False,
            chunk_size=0,
            num_chunks=0,
            mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
        ),
        benchmark_mla.CaseReport(
            case=benchmark_mla.DecodeCase(mode="prefill", batch_size=2, cache_len=32768, topk=2048, q_len=16384),
            graph_width=32768,
            metadata_us=400.0,
            replay_us=100.0,
            indexer_us=400.0,
            mla_us=100.0,
            split_enabled=True,
            chunk_size=256,
            num_chunks=8,
            mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
        ),
    ]

    lines = benchmark_mla._render_summary_lines(reports)

    assert lines[0] == "Summary"
    assert lines[1] == "  cases: 2"
    assert lines[2] == "  total geo:   500.00 us"
    assert lines[3] == "  indexer geo: 200.00 us"
    assert lines[4] == "  mla geo:     200.00 us"
    assert lines[5] == "  step geo:    500.00 us"
    assert lines[6] == "  meta geo:    200.00 us"
    assert lines[7] == "  replay geo:  200.00 us"


def test_main_prints_no_stdout_on_failure(monkeypatch, capsys) -> None:
    case = benchmark_mla.DecodeCase(mode="decode", batch_size=1, cache_len=1024, topk=1024)

    def fake_collect_case_reports(args, *, device=None):
        del args, device
        raise benchmark_mla.BenchmarkFailure(case, "synthetic failure")

    monkeypatch.setattr(benchmark_mla, "collect_case_reports", fake_collect_case_reports)

    rc = benchmark_mla.main([])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "synthetic failure" in captured.err


def test_main_prints_buffered_case_lines_and_summary(monkeypatch, capsys) -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(mode="prefill", batch_size=1, cache_len=32768, topk=2048, q_len=16384),
        graph_width=8192,
        metadata_us=11.0,
        replay_us=322.0,
        indexer_us=111.0,
        mla_us=222.0,
        split_enabled=True,
        chunk_size=128,
        num_chunks=8,
        mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
    )

    def fake_collect_case_reports(args, *, device=None):
        del args, device
        return [report]

    monkeypatch.setattr(benchmark_mla, "collect_case_reports", fake_collect_case_reports)

    rc = benchmark_mla.main([])

    captured = capsys.readouterr()
    assert rc == 0
    assert "glm51-prefill tp8" in captured.out
    assert "Summary" in captured.out
    assert captured.err == ""
