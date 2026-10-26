import pytest
from src.schemas import ClinicalQueryRequest
from src.rag_engine import ClinicalRAGEngine

@pytest.fixture
def rag_engine():
    return ClinicalRAGEngine()

def test_sepsis_norepinephrine_query(rag_engine):
    req = ClinicalQueryRequest(
        query_text="vasopressor primeira escolha choque septico",
        top_k_retrieval=2,
        similarity_threshold=0.10
    )
    res = rag_engine.answer_query(req)
    assert len(res.grounded_sources) > 0
    assert "SSC-2021-03" in [p.guideline_id for p in res.grounded_sources]
    assert "Norepinefrina" in res.synthesized_answer
    assert res.faithfulness_score > 0.80

def test_kdigo_aki_query(rag_engine):
    req = ClinicalQueryRequest(
        query_text="definicao estadiamento injuria renal aguda creatinina KDIGO",
        top_k_retrieval=1,
        similarity_threshold=0.10
    )
    res = rag_engine.answer_query(req)
    assert len(res.grounded_sources) == 1
    assert res.grounded_sources[0].guideline_id == "KDIGO-2012-AKI"
    assert res.hallucination_risk in ["LOW", "MEDIUM"]

def test_irrelevant_query_handling(rag_engine):
    req = ClinicalQueryRequest(
        query_text="qual a receita de bolo de chocolate com morango",
        top_k_retrieval=3,
        similarity_threshold=0.50
    )
    res = rag_engine.answer_query(req)
    assert len(res.grounded_sources) == 0
    assert res.hallucination_risk == "HIGH"
