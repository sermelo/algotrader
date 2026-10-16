import pandas as pd
import pytest

from algotrader.run import main
from helpers import make_bars

UP_THEN_DOWN = [10] * 5 + [11, 12, 13, 14, 15] + [14.5, 14, 13.5, 13, 12.5, 12]


@pytest.fixture
def data_dir(tmp_path):
    bars = make_bars(UP_THEN_DOWN * 3)
    raw = bars.rename(columns=str.capitalize).rename_axis("datetime").reset_index()
    raw.to_parquet(tmp_path / "ABC_clean.parquet")
    return tmp_path


def test_cli_runs_all_tickers_and_writes_csv(data_dir, capsys):
    out = data_dir / "res" / "summary.csv"
    summary = main(["--data-dir", str(data_dir), "--timeframe", "1min", "--fast", "2",
                    "--slow", "3", "--atr-period", "2", "--min-commission", "0",
                    "--commission", "0", "--out", str(out)])
    assert summary["ticker"].tolist() == ["ABC"]
    assert summary.loc[0, "trades"] > 0 and summary.loc[0, "total_commission"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(out), summary, check_dtype=False)
    assert "ABC:" in capsys.readouterr().out


def test_cli_tree_fits_before_train_end_and_tests_after(tmp_path):
    zigzag = [100 + 0.01 * i + (0.5 if i % 2 else -0.5) for i in range(400)]
    raw = make_bars(zigzag).rename(columns=str.capitalize).rename_axis("datetime").reset_index()
    raw.to_parquet(tmp_path / "ZZ_clean.parquet")
    args = ["--data-dir", str(tmp_path), "--timeframe", "1min", "--strategy", "tree",
            "--atr-mult", "50", "--commission", "0", "--min-commission", "0"]
    summary = main(args + ["--train-end", "2024-01-02 12:50"])   # 200 bars each side
    assert summary.loc[0, "trades"] > 50
    with pytest.raises(SystemExit):                               # --train-end required
        main(args)
    with pytest.raises(SystemExit):                               # nothing left to test
        main(args + ["--train-end", "2030-01-01"])


def test_cli_rejects_empty_folder_and_bad_windows(tmp_path, data_dir):
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path / "nothing")])
    with pytest.raises(SystemExit):
        main(["--data-dir", str(data_dir), "--fast", "50", "--slow", "20"])
