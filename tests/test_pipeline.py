from creditforyou.model import train
from creditforyou.pipeline import evaluate
from creditforyou.report import kpi_summary
from creditforyou.sample import generate


def test_end_to_end(tmp_path):
    generate(n_customers=40, n_loans=3000, seed=1, out_dir=tmp_path)
    model_path = tmp_path / "pd_model.joblib"
    art = train(tmp_path / "loans_training.csv", model_path=model_path, log=lambda *_: None)
    # không được dùng cột rò rỉ của Lending Club
    feats = art["num_features"] + art["cat_features"]
    assert not {"int_rate", "grade", "total_pymnt", "installment"} & set(feats)

    res, warnings, model = evaluate(tmp_path / "transactions.csv", tmp_path / "applicants.csv",
                                    model_path=model_path)
    assert model is not None and len(res) == 40
    assert res["credit_score"].between(0, 100).all()
    assert set(res["tier"]) <= set("ABCD")
    assert res["pd"].between(0, 1).all()
    k = kpi_summary(res, 1.0)
    assert "AUC_diem_tin_dung" in k and "thu_nhap_MAPE" in k


def test_without_model_and_applicants(tmp_path):
    generate(n_customers=5, n_loans=100, seed=2, out_dir=tmp_path)
    res, warnings, model = evaluate(tmp_path / "transactions.csv", model_path=tmp_path / "missing.joblib",
                                    overrides={"requested_amount": 30e6})
    assert model is None and len(res) == 5
    assert (res["pd_used"] == 0.5).all()
    assert any("Chưa có mô hình" in w for w in warnings)
