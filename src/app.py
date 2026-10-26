import streamlit as st
import pandas as pd
from src.schemas import ClinicalQueryRequest
from src.rag_engine import ClinicalRAGEngine

st.set_page_config(
    page_title="Clinical RAG Medical Guidelines Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Clinical RAG: Assistente Baseado em Evidências Médicas")
st.markdown("""
Pipeline de **Recuperação Aumentada por Geração (RAG)** com controle estrito de contexto, rastreabilidade de passagens e eliminação de alucinações clínicas.
""")

engine = ClinicalRAGEngine()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Parâmetros de Recuperação")
    sample_queries = [
        "Qual o vasopressor de primeira linha no choque séptico?",
        "Qual a meta de PAM recomendada na sepse com vasopressores?",
        "Qual a definição de injúria renal aguda de acordo com KDIGO?",
        "Qual a dose inicial de cristalóides na ressuscitação de sepse?"
    ]
    selected_sample = st.selectbox("Perguntas clínicas frequentes:", sample_queries)
    query_text = st.text_area("Pergunta Clínica:", value=selected_sample, height=100)
    top_k = st.slider("Top-K Passagens para Contexto", 1, 5, 3)
    min_sim = st.slider("Limiar Mínimo de Similaridade Coseno", 0.05, 0.50, 0.15, 0.05)

with col2:
    st.subheader("📋 Resposta Sintetizada com Citação de Diretrizes")
    req = ClinicalQueryRequest(
        query_text=query_text,
        top_k_retrieval=top_k,
        similarity_threshold=min_sim
    )
    res = engine.answer_query(req)

    badge_color = "🟢" if res.hallucination_risk == "LOW" else ("🟡" if res.hallucination_risk == "MEDIUM" else "🔴")
    st.info(f"**Score de Fidelidade**: `{res.faithfulness_score:.2%}` | **Risco de Alucinação**: {badge_color} `{res.hallucination_risk}`")

    st.markdown(f"### 💡 Resposta Clínica Grounded:")
    st.write(res.synthesized_answer)

    st.caption(f"⚠️ *{res.disclaimer}*")

    st.markdown("### 📚 Diretrizes e Passagens Recuperadas (Ground Truth)")
    if res.grounded_sources:
        passages_data = []
        for p in res.grounded_sources:
            passages_data.append({
                "ID": p.guideline_id,
                "Diretriz": p.title,
                "Seção": p.section,
                "Nível de Evidência": p.evidence_level,
                "Similaridade Coseno": f"{p.cosine_similarity:.4f}",
                "Trecho Extraído": p.content
            })
        st.dataframe(pd.DataFrame(passages_data), use_container_width=True)
    else:
        st.warning("Nenhuma passagem médica atendeu ao limiar de similaridade.")
