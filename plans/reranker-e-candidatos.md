# Plano — Reranker + pool maior de candidatos

- **Status:** proposto (não implementado)
- **Data:** 2026-11-17
- **Relacionado:** item 1 (gate de relevância + reranker) das melhorias inspiradas no Google
- **Pré-requisitos:** nenhuma reindexação; o índice atual continua válido

## 1. Objetivo

Melhorar a precisão do contexto entregue ao LLM em duas etapas:

1. **Aumentar o recall**: buscar um pool maior de candidatos (ex.: 24) usando a
   busca híbrida + fan-out atual.
2. **Aumentar a precisão**: reaproveitar um reranker (cross-encoder ou API) para
   reordenar esse pool e aplicar um limiar de relevância antes de escolher os
   `k` trechos finais.

Isso ataca dois problemas visíveis hoje:

- perguntas sem sentido recuperam trechos só porque o FAISS sempre devolve
  vizinhos próximos;
- com `k` pequeno, um bom trecho pode ficar de fora do topo do RRF.

## 2. Não-objetivos

- Não trocar o FAISS por banco vetorial.
- Não reindexar nem mudar chunking/modelo de embedding.
- Não implementar gate/limiar baseado em cosseno do embedding (fica como
  alternativa se o reranker não for adotado).
- Não paralelizar as buscas do fan-out.
- Não implementar multi-tenant, API ou cache.

## 3. Estado atual

- `src/rag_poc/retriever.py`
  - `Retriever.retrieve_many(queries, per_query_k=None)` faz RRF por consulta
    (FAISS + BM25) e outro RRF entre consultas.
  - O resultado é truncado em `Settings.retrieval_k` (default 4; no `.env`
    local está 5).
  - Cada consulta usa `per_query_k = max(retrieval_k, fanout_candidates_per_query)`
    (default 8).
- `src/rag_poc/cli.py`
  - Fluxo: planner/fan-out → `retrieve_many` → loop de suficiência →
    `chain.stream` → fontes.
- `src/rag_poc/chain.py`
  - Planner devolve 1..N consultas; `queries[0]` é a pergunta principal
    reescrita.
  - Suficiência pode adicionar consultas de lacuna e refazer a busca.
- `src/rag_poc/ui.py`
  - Tabela de fontes e citações linkadas; ainda não exibe scores.
- **Não existe reranker** hoje.

## 4. Arquitetura alvo

```text
pergunta
  │
  ▼
planner (1..N consultas)
  │
  ▼
retrieve_many(pool_limit)          ← recall: até RERANK_CANDIDATES
  │  (FAISS + BM25 + RRF por consulta + RRF entre consultas)
  ▼
reranker(query=queries[0], pool)   ← precisão: score por par
  │
  ▼
limiar (RERANK_MIN_SCORE) + top k  ← gate de relevância
  │
  ├── vazio ──► aviso + resposta com conhecimento geral
  │
  ▼
suficiência? ── não ──► + consultas de lacuna
  │                     └─► retrieve_many(pool) → reranker → top k
  ▼
prompt → LLM → resposta + fontes
```

Ordem importante: **rerankar antes da suficiência**. No loop de lacuna, o pool
é recalculado com as consultas ampliadas e passa pelo reranker novamente.

## 5. Decisões propostas

| Decisão | Proposta | Motivo |
|---|---|---|
| Tamanho do pool | `RERANK_CANDIDATES=24` | Bom equilíbrio custo/recall para 5–6k chunks |
| Corte final | `RETRIEVAL_K` atual (4/5) | Não muda o custo do prompt |
| Provider inicial | `openrouter` (`voyageai/rerank-3-lite`) | Mesma chave, sem torch, score 0–1, ~US$ 0,0001/consulta |
| Ativação | `RERANK_ENABLED=false` até calibrar | Evita mudança de comportamento antes do A/B |
| Limiar inicial | `RERANK_MIN_SCORE=0.0` | Só reordena; calibração vem depois |
| Query do reranker | `queries[0]` (pergunta principal reescrita) | Evita rerankar com follow-up elíptico |
| Falha do reranker | Fallback para ordem do RRF | Não bloqueia o turno |
| Deduplicação | por `doc_id` antes do rerank | Evita gastar com duplicatas |
| Texto enviado | até `RERANK_TEXT_CHARS=2000` por chunk | Controla custo/limite do provider |
| Provider local | fase 2 (`flashrank`) | Opção offline/privada |

## 6. Configuração nova

Adicionar em `Settings` (`src/rag_poc/config.py`) e `.env.example`:

```dotenv
RERANK_ENABLED=false
RERANK_PROVIDER=openrouter        # openrouter | local
RERANK_MODEL=voyageai/rerank-3-lite
RERANK_CANDIDATES=24
RERANK_MIN_SCORE=0.0
RERANK_TEXT_CHARS=2000
RERANK_TIMEOUT=20
```

Validações:

- `RERANK_PROVIDER ∈ {openrouter, local}`
- `RERANK_CANDIDATES >= RETRIEVAL_K`
- `RERANK_MIN_SCORE` entre 0 e 1
- `RERANK_TEXT_CHARS >= 100`
- `RERANK_TIMEOUT >= 1`

## 7. Mudanças por arquivo

### 7.1 `src/rag_poc/retriever.py`

- Adicionar limite de pool sem mudar o comportamento default:
  ```python
  def retrieve_many(
      self,
      queries: Sequence[str],
      *,
      per_query_k: int | None = None,
      limit: int | None = None,
  ) -> list[RetrievedChunk]: ...
  ```
  - `limit` default = `settings.retrieval_k` (comportamento atual).
  - Com reranker ligado, a CLI passa `limit=settings.rerank_candidates`.
- Cálculo de `per_query_k` quando reranker ligado:
  ```text
  per_query_k = max(
      fanout_candidates_per_query,
      ceil(rerank_candidates / len(queries))
  )
  ```
  Exemplo: 24 candidatos e 3 consultas → 8 por consulta; 1 consulta → 24.
- Preservar deduplicação por `doc_id` e proveniência (`queries`, `sources`).
- Nenhuma mudança no índice.

### 7.2 Novo `src/rag_poc/reranker.py`

- Estender `RetrievedChunk` com campo opcional:
  ```python
  rerank_score: float | None = None
  ```
  (default mantém compatibilidade com todos os testes/constructos atuais).
- Interface:
  ```python
  class Reranker(Protocol):
      def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]: ...
  ```
- `OpenRouterReranker`:
  - `POST https://openrouter.ai/api/v1/rerank`
  - body:
    ```json
    {
      "model": "voyageai/rerank-3-lite",
      "query": "...",
      "documents": ["texto 1", "texto 2"],
      "top_n": 24
    }
    ```
  - headers: `Authorization`, `HTTP-Referer`, `X-Title` (reaproveitar config).
  - resposta: `results[*].index` + `results[*].relevance_score`.
  - usar `httpx` (já é dependência transitiva do `openai`; declarar direto).
  - timeout e no máximo 2 retries; erro → `RerankerError`.
  - truncar cada documento em `RERANK_TEXT_CHARS`.
- `build_reranker(settings) -> Reranker | None`:
  - `None` se `RERANK_ENABLED=false`.
  - `openrouter` → `OpenRouterReranker`.
  - `local` → `LocalFlashRankReranker` (fase 2; erro claro se indisponível).
- Função pura `apply_rerank_order(chunks, results) -> list[RetrievedChunk]` para
  facilitar testes: aplica scores, ordena, descarta índices inválidos e mantém
  fallback.
- Exceção `RerankerError(RagError)`.

### 7.3 `src/rag_poc/cli.py`

Fluxo dentro do `try` do turno:

1. Planner → `queries`.
2. `pool = retriever.retrieve_many(queries, limit=rerank_candidates)`
   quando reranker ligado; senão `retrieve_many(queries)`.
3. `reranker = build_reranker(settings)` (construir uma vez no início do chat).
4. Se reranker:
   - `reranked = reranker.rerank(queries[0], pool)`;
   - filtrar `rerank_score >= RERANK_MIN_SCORE`;
   - `chunks = filtered[:retrieval_k]`.
5. Se o filtro esvaziar:
   - `ui.print_rerank_gate(dropped_count, threshold)`;
   - seguir para o caminho de conhecimento geral (sem suficiência).
6. Suficiência no `chunks` final; se adicionar consultas:
   - `queries = merge_queries(...)`;
   - recalcular pool **da mesma forma** e rerankar de novo;
   - respeitar `SUFFICIENCY_MAX_ROUNDS`.
7. Falha do reranker (`RerankerError`/timeout):
   - `ui.print_reranker_fallback()`;
   - usar o pool na ordem do RRF, sem limiar.
8. `ui.stream_answer(..., chunks=chunks, link_scheme=...)` sem mudanças.

Construir o reranker e o cliente HTTP uma única vez por sessão; reutilizar entre
turnos.

### 7.4 `src/rag_poc/ui.py`

- Novo `--show-scores` (ou estender `--show-chunks`):
  - mostrar posição no pool, score do RRF, score do reranker e indicador
    `mantido/descartado`.
- Nova coluna opcional “Rerank” na tabela de fontes quando houver score.
- Mensagens novas:
  - `print_rerank_gate(dropped, threshold)`
  - `print_reranker_fallback()`
- Banner: `Rerank: voyageai/rerank-3-lite · pool 24` ou `desligado`.

### 7.5 `src/rag_poc/chain.py`

- Nenhuma mudança obrigatória.
- A query do reranker é `queries[0]`; se o fan-out estiver desligado e houver
  histórico, `queries[0]` continua sendo a pergunta condensada.

### 7.6 Testes

Novos testes com reranker falso (sem API):

- `retriever.retrieve_many(..., limit=24)` devolve até 24 candidatos e mantém
  dedupe/proveniência.
- Reranker fake reordena, preenche `rerank_score` e a CLI usa os `k` melhores.
- Limiar filtra corretamente; se tudo cair, cai no caminho de conhecimento
  geral e mostra aviso.
- Falha do reranker usa ordem do RRF e não interrompe a resposta.
- Loop de suficiência reranka o pool após adicionar consultas de lacuna
  (verificar duas chamadas ao reranker).
- `RERANK_ENABLED=false` mantém exatamente o comportamento atual (103 testes
  existentes passando).
- Parsing/resposta do provider OpenRouter com JSON mockado: índices fora de
  faixa, scores ausentes, lista vazia.
- Config: validações de provider, pool, limiar, timeout.

### 7.7 Documentação

- `README.md`: seção “Reranker e candidatos”, configurações novas, custo/latência,
  como desligar.
- `docs/ARQUITETURA.md`: diagrama e seção do fluxo com reranker; glossário.
- `.env.example`: bloco de configuração.
- `plans/` já contém este documento.

## 8. Calibração do limiar

Não ligar `RERANK_MIN_SCORE` sem medir. Procedimento:

1. Criar um conjunto de 15–20 perguntas:
   - 8–10 reais sobre React/Vite;
   - 3–5 follow-ups;
   - 4–5 sem sentido.
2. Rodar com `RERANK_ENABLED=true`, `RERANK_MIN_SCORE=0.0` e `--show-scores`.
3. Medir:
   - recall@24 (o trecho certo está no pool?);
   - precision@k após rerank;
   - falsos negativos (bons trechos descartados pelo limiar);
   - latência e custo.
4. Testar limiares candidatos (0,1 / 0,2 / 0,3 / 0,4) e escolher o que reduz
   ruído sem derrubar boas respostas.
5. Só então mudar o default para o valor calibrado.

Referência de qualidade: no benchmark público do OpenRouter, o top-1 passou de
81% (só embeddings) para ~97% com `voyageai/rerank-3-lite`, a ~US$ 0,10 por
1.000 consultas.

## 9. Custo e latência

| Item | Custo aproximado | Latência |
|---|---|---|
| Pool 24 + Voyage lite | ~US$ 0,0001/turno | +0,3–1,5 s |
| Pool 24 + FlashRank local | zero | +50–150 ms (CPU) |
| LLM reranker (opcional) | ~US$ 0,0007/turno | +1–3 s |
| Resposta | inalterado (`k` chunks) | inalterado |

## 10. Riscos e mitigações

- **Limiar alto demais**: falsos negativos; calibrar e começar em 0,0.
- **Provider indisponível/429**: fallback para RRF + aviso; nunca quebrar o turno.
- **Score não comparável entre modelos**: default por provider; documentar.
- **Privacidade**: provider remoto recebe o texto dos chunks; opção local.
- **Custo/latência**: pool limitado; truncar documentos; medir p95.
- **Interação com suficiência**: rerank antes; após lacuna, rerankar o pool novo.
- **Redundância**: dedupe por `doc_id` antes do rerank; opcional limitar chunks
  por arquivo (diversidade) numa fase futura.

## 11. Fases de implementação

1. **Fase 1 — Pool maior, sem reranker**
   - `limit` no `retrieve_many` + `--show-scores`.
   - Medir recall@24 e latência. Nenhuma mudança de resposta.
2. **Fase 2 — Reranker OpenRouter**
   - `reranker.py`, config, integração CLI/UI, testes com fake.
   - `RERANK_ENABLED=false` por default.
3. **Fase 3 — Calibração**
   - Conjunto de avaliação, escolha do limiar e do `RERANK_CANDIDATES`.
   - A/B com/sem reranker.
4. **Fase 4 — Opcional**
   - Provider local `flashrank`.
   - Coluna de diversidade/máx. chunks por arquivo.

## 12. Critérios de aceite

- Com `RERANK_ENABLED=false`, o comportamento e os 103 testes atuais não mudam.
- Com reranker fake: pool de até 24, ordenação por score, limiar aplicado,
  `k` final respeitado, fallback em erro.
- Com OpenRouter real: 24 candidatos rerankeados, latência adicional ≤ 2 s,
  nenhuma chave exposta em logs.
- Loop de suficiência reranka o pool ampliado e respeita o número máximo de
  rodadas.
- `--show-scores` permite ver claramente o antes/depois.
- Nenhuma reindexação necessária.
- `ruff check`, `ruff format --check` e `pytest` verdes.

## 13. Decisões em aberto

1. Provider default: OpenRouter (recomendado) ou FlashRank local?
2. `RERANK_CANDIDATES`: 24 ou 30?
3. Limiar inicial após calibração: 0,2 ou 0,3?
4. `RERANK_ENABLED` default: `false` até o A/B (recomendado) ou `true`?
5. `--show-scores` separado ou incorporado ao `--show-chunks`?
6. Implementar FlashRank já na fase 2 ou deixar para a fase 4?
