# CRISP-DM: Entendimento do Negócio - RAG Clínico Seguro

## 1. Problema Clínico
Modelos de Linguagem Genéricos (LLMs) sofrem de alucinações factuais graves quando consultados sobre dosagens de antibióticos, metas hemodinâmicas em choque séptico ou estadiamento de injúria renal aguda (IRA).
No ambiente de terapia intensiva e emergência, o erro de dosagem ou diretriz pode ser fatal.

## 2. Estratégia de Mitigação (RAG com Citação Estrita)
- **Recuperação Semântica Precisa**: Indexação em chunks estruturados por protocolo clínico.
- **Reranking por Nível de Evidência**: Priorização de recomendações com grau GRADE 1A/1B.
- **Auditoria de Fidelidade**: Cálculo de sobreposição léxica e semântica entre a resposta gerada e os trechos recuperados.
