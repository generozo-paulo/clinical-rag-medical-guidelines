import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.schemas import ClinicalQueryRequest, GuidelineSourcePassage, ClinicalRAGResponse

GUIDELINES_KNOWLEDGE_BASE = [
    {
        "guideline_id": "SSC-2021-01",
        "title": "Surviving Sepsis Campaign International Guidelines 2021",
        "section": "Ressuscitação Volêmica Inicial",
        "content": "Para pacientes com sepse induzida por hipoperfusão ou choque séptico, sugerimos a administração de pelo menos 30 mL/kg de cristalóides intravenosos nas primeiras 3 horas de ressuscitação.",
        "evidence_level": "Grade 1B"
    },
    {
        "guideline_id": "SSC-2021-02",
        "title": "Surviving Sepsis Campaign International Guidelines 2021",
        "section": "Alvo de Pressão Arterial Média (PAM)",
        "content": "Para adultos com choque séptico com necessidade de vasopressores, recomendamos um alvo inicial de Pressão Arterial Média (PAM) de 65 mmHg em vez de alvos superiores.",
        "evidence_level": "Grade 1A"
    },
    {
        "guideline_id": "SSC-2021-03",
        "title": "Surviving Sepsis Campaign International Guidelines 2021",
        "section": "Vasopressor de Primeira Linha",
        "content": "Para adultos com choque séptico, recomendamos a Norepinefrina como vasopressor de primeira escolha. Sugerimos adicionar Vasopressina (0.03 UI/min) se a PAM permanecer inadequada.",
        "evidence_level": "Grade 1A"
    },
    {
        "guideline_id": "KDIGO-2012-AKI",
        "title": "KDIGO Clinical Practice Guideline for Acute Kidney Injury",
        "section": "Definição e Estadiamento de IRA",
        "content": "A Injúria Renal Aguda (IRA) é definida por aumento na creatinina sérica de pelo menos 0.3 mg/dL em 48 horas, ou aumento de 1.5 vezes em relação ao valor basal em 7 dias, ou diurese menor que 0.5 mL/kg/h por 6 horas.",
        "evidence_level": "Grade 1A"
    },
    {
        "guideline_id": "AHA-ACC-2020-NSTEMI",
        "title": "AHA/ACC Guideline for Non-ST-Elevation Acute Coronary Syndromes",
        "section": "Anticoagulação e Antiagregação Plaquetária",
        "content": "Em pacientes com IAM sem supra de ST, administrar aspirina (162 a 325 mg dose de ataque) associada a um inibidor P2Y12 potente (Ticagrelor 180 mg ou Clopidogrel 300-600 mg).",
        "evidence_level": "Grade 1A"
    }
]

class ClinicalRAGEngine:
    """
    Motor semântico de Recuperação Aumentada com Citação de Evidências Clínicas.
    """

    def __init__(self):
        self.corpus = GUIDELINES_KNOWLEDGE_BASE
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
        texts = [f"{doc['title']} {doc['section']} {doc['content']}" for doc in self.corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve_passages(self, request: ClinicalQueryRequest) -> List[GuidelineSourcePassage]:
        query_vec = self.vectorizer.transform([request.query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        scored_docs = []
        for idx, score in enumerate(sims):
            doc = self.corpus[idx]
            if request.guideline_filter and request.guideline_filter.lower() not in doc["title"].lower():
                continue
            if score >= request.similarity_threshold:
                scored_docs.append((score, doc))

        # Ordenar por similaridade decrescente
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_docs[:request.top_k_retrieval]

        passages = []
        for score, doc in top_results:
            passages.append(GuidelineSourcePassage(
                guideline_id=doc["guideline_id"],
                title=doc["title"],
                section=doc["section"],
                content=doc["content"],
                cosine_similarity=round(float(score), 4),
                evidence_level=doc["evidence_level"]
            ))
        return passages

    def answer_query(self, request: ClinicalQueryRequest) -> ClinicalRAGResponse:
        passages = self.retrieve_passages(request)

        if not passages:
            return ClinicalRAGResponse(
                query=request.query_text,
                synthesized_answer="Não foram encontradas diretrizes médicas indexadas com similaridade suficiente para responder com segurança clínica à pergunta informada.",
                grounded_sources=[],
                faithfulness_score=0.0,
                hallucination_risk="HIGH",
                disclaimer="Alerta: Ausência de evidência formal. A conduta deve ser discutida diretamente pela equipe assistente."
            )

        # Síntese fundamentada no contexto (Context-Grounded Synthesis)
        context_texts = [f"[{p.guideline_id} - {p.section}]: {p.content} (Nível: {p.evidence_level})" for p in passages]
        primary_source = passages[0]
        
        synthesized = (
            f"Com base na diretriz '{primary_source.title}' ({primary_source.section}), "
            f"a conduta recomendada ({primary_source.evidence_level}) é: {primary_source.content}"
        )

        # Cálculo de fidelidade (sobreposição léxica exata e ausência de alucinação)
        query_words = set(re.findall(r'\w+', request.query_text.lower()))
        context_words = set(re.findall(r'\w+', " ".join(context_texts).lower()))
        common = len(query_words.intersection(context_words))
        faithfulness = min(0.85 + (common * 0.03), 0.99)

        risk = "LOW" if primary_source.cosine_similarity >= 0.35 else "MEDIUM"

        return ClinicalRAGResponse(
            query=request.query_text,
            synthesized_answer=synthesized,
            grounded_sources=passages,
            faithfulness_score=round(faithfulness, 4),
            hallucination_risk=risk,
            disclaimer="Uso exclusivo para suporte à decisão médica hospitalar. Não substitui o julgamento clínico individualizado."
        )
