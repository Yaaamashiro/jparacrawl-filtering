import subprocess
from pathlib import Path

import pytest

from jparacrawl_filtering.scoring import BicleanerScorer, cuda_available


def test_cpu_adapter_and_direction(config, tmp_path, monkeypatch):
    c = config["bicleaner"]
    c["device"] = "cpu"
    calls = []
    monkeypatch.setattr(
        "jparacrawl_filtering.scoring.shutil.which", lambda _: "bicleaner-ai-classify"
    )

    def cli(command, **kw):
        calls.append(command)
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert Path(command[1]).read_text(encoding="utf-8") == "English.\t日本語。\n"
        assert "--disable_hardrules" in command and "--score_only" in command
        Path(command[2]).write_text("0.876\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("jparacrawl_filtering.scoring.subprocess.run", cli)
    scorer = BicleanerScorer(c, tmp_path)
    assert scorer.score([{"en_norm": "English.", "ja_norm": "日本語。"}]) == [0.876]
    assert len(calls) == 1 and not list(tmp_path.glob("*.tsv"))


def test_oom_retries_then_cpu_fallback(config, tmp_path, monkeypatch):
    c = config["bicleaner"]
    monkeypatch.setattr("jparacrawl_filtering.scoring.cuda_available", lambda: True)
    monkeypatch.setattr(
        "jparacrawl_filtering.scoring.shutil.which", lambda _: "bicleaner-ai-classify"
    )
    batches = []

    def cli(command, **kw):
        b = int(command[command.index("--batch_size") + 1])
        batches.append(b)
        if "--require_gpu" in command:
            return subprocess.CompletedProcess(command, 1, "", "ResourceExhausted: out of memory")
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        Path(command[2]).write_text("0.9\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("jparacrawl_filtering.scoring.subprocess.run", cli)
    scorer = BicleanerScorer(c, tmp_path)
    assert scorer.score([{"en_norm": "English.", "ja_norm": "日本語。"}]) == [0.9]
    assert batches == [32, 16, 8, 4, 2, 1, 1]
    assert scorer.device == "cpu"


@pytest.mark.parametrize("output", ["nan\n", "1.1\n", "bad\n", "", "0.5\n0.6\n"])
def test_bad_scores_rejected(config, tmp_path, monkeypatch, output):
    config["bicleaner"]["device"] = "cpu"
    monkeypatch.setattr(
        "jparacrawl_filtering.scoring.shutil.which", lambda _: "bicleaner-ai-classify"
    )

    def cli(command, **kw):
        Path(command[2]).write_text(output, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("jparacrawl_filtering.scoring.subprocess.run", cli)
    with pytest.raises(ValueError):
        BicleanerScorer(config["bicleaner"], tmp_path).score(
            [{"en_norm": "English.", "ja_norm": "日本語。"}]
        )


def test_cuda_unavailable(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert not cuda_available()


def test_timeout_keeps_checkpoint_contract(config, tmp_path, monkeypatch):
    config["bicleaner"]["device"] = "cpu"
    monkeypatch.setattr(
        "jparacrawl_filtering.scoring.shutil.which", lambda _: "bicleaner-ai-classify"
    )

    def cli(command, **kw):
        raise subprocess.TimeoutExpired(command, 1, stderr="sensitive synthetic sentence")

    monkeypatch.setattr("jparacrawl_filtering.scoring.subprocess.run", cli)
    with pytest.raises(RuntimeError, match="completed shards are safe") as error:
        BicleanerScorer(config["bicleaner"], tmp_path).score(
            [{"en_norm": "English.", "ja_norm": "日本語。"}]
        )
    assert "sensitive" not in str(error.value)
    assert not list(tmp_path.glob("inference-*"))
