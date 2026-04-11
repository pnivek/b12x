from __future__ import annotations

from benchmarks import benchmark_mla


def test_build_decode_cases_covers_requested_matrix_and_topk_cap() -> None:
    cases = benchmark_mla._build_decode_cases(
        batch_sizes=[1, 2, 4, 8],
        cache_lens=[1024, 8192, 65536],
        topk_cap=2048,
    )

    assert len(cases) == 12
    assert cases[0] == benchmark_mla.DecodeCase(batch_size=1, cache_len=1024, topk=1024)
    assert cases[1] == benchmark_mla.DecodeCase(batch_size=1, cache_len=8192, topk=2048)
    assert cases[-1] == benchmark_mla.DecodeCase(batch_size=8, cache_len=65536, topk=2048)


def test_render_case_line_reports_public_step_metrics() -> None:
    report = benchmark_mla.CaseReport(
        case=benchmark_mla.DecodeCase(batch_size=4, cache_len=8192, topk=2048),
        indexer_us=123.45,
        mla_us=456.78,
        split_enabled=True,
        chunk_size=256,
        num_chunks=8,
        mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
    )

    line = benchmark_mla._render_case_line(report)

    assert "glm51-decode tp8" in line
    assert "bs= 4" in line
    assert "ctx=  8192" in line
    assert "topk=2048" in line
    assert "split= on" in line
    assert "chunk=256" in line
    assert "nchunks=8" in line
    assert "total=" in line
    assert "indexer=" in line
    assert "mla=" in line


def test_render_summary_lines_reports_geomeans() -> None:
    reports = [
        benchmark_mla.CaseReport(
            case=benchmark_mla.DecodeCase(batch_size=1, cache_len=1024, topk=1024),
            indexer_us=100.0,
            mla_us=400.0,
            split_enabled=False,
            chunk_size=0,
            num_chunks=0,
            mla_sanity=benchmark_mla.SanityMetrics(max_abs=0.01, rmse=0.001, cos=0.9999),
        ),
        benchmark_mla.CaseReport(
            case=benchmark_mla.DecodeCase(batch_size=2, cache_len=8192, topk=2048),
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


def test_main_prints_no_stdout_on_failure(monkeypatch, capsys) -> None:
    case = benchmark_mla.DecodeCase(batch_size=1, cache_len=1024, topk=1024)

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
        case=benchmark_mla.DecodeCase(batch_size=1, cache_len=1024, topk=1024),
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
    assert "glm51-decode tp8" in captured.out
    assert "Summary" in captured.out
    assert captured.err == ""
