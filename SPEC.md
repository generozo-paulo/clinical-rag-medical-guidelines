# Especificação Contratual (SDD) - Clinical RAG Medical Guidelines

## 1. Escopo & Objetivos
Pipeline de Recuperação Aumentada por Geração (**RAG**) com embeddings semânticos, reranking e citação estrita de diretrizes médicas internacionais (Surviving Sepsis Campaign, Diretrizes AHA/ACC e KDIGO). O sistema mitiga alucinações através de controle estrito de contexto e medição de fidelidade (Faithfulness & Answer Relevance).

## 2. Contratos de Entrada (Pydantic)

### 2.1 Pergunta Clínica (`ClinicalQueryRequest`)
- `query_text`: str
- `top_k_retrieval`: int = Field(default=3, ge=1, le=10)
- `similarity_threshold`: float = Field(default=0.45, ge=0.0, le=1.0)
- `guideline_filter`: Optional[str] = None

## 3. Contratos de Saída (Pydantic)

### 3.1 Fonte Citada (`GuidelineSourcePassage`)
- `guideline_id`: str
- `title`: str
- `section`: str
- `content`: str
- `cosine_similarity`: float
- `evidence_level`: Literal["Grade 1A", "Grade 1B", "Grade 2C", "Expert Consensus"]

### 3.2 Resposta Gerada Auditada (`ClinicalRAGResponse`)
- `query`: str
- `synthesized_answer`: str
- `grounded_sources`: List[GuidelineSourcePassage]
- `faithfulness_score`: float
- `hallucination_risk`: Literal["LOW", "MEDIUM", "HIGH"]
- `disclaimer`: str
