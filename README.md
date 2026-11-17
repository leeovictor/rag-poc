# RAG Híbrido (PoC)

CLI de RAG que indexa PDFs, TXTs e Markdown/MDX em um índice FAISS local,
combina busca vetorial com BM25 e responde perguntas citando as fontes.
Embeddings e LLM usam uma única chave do OpenRouter.

A indexação é otimizada para inglês; as respostas saem em português.

## Recursos

- `ingest`: indexa arquivo ou diretório, com chunking, dedupe e índice local.
- `ingest --dry-run`: estima chunks, tokens, custo e tempo sem chamar a API.
- `chat`: streaming, memória, fan-out de consultas, loop de suficiência,
  citações `[n]` clicáveis e tabela de fontes.
- `status` / `clear`: inspeciona ou remove o índice.

## Requisitos

- Python 3.10+
- Chave do [OpenRouter](https://openrouter.ai/keys)

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # opcional: habilita o comando `rag`

cp .env.example .env
# edite .env e preencha OPENROUTER_API_KEY
```

## Uso

```bash
rag ingest ./docs --dry-run       # estimativa, sem gastar
rag ingest ./docs                 # indexa
rag ingest ./docs --rebuild       # reindexa do zero

rag chat                          # conversa
rag chat --k 6 --show-chunks      # mais trechos + preview
rag status
rag clear --yes
```

Sem `pip install -e .`, use `PYTHONPATH=src python -m rag_poc ...` (ou
`.venv/bin/rag ...`).

No chat: `sair`, `exit`, `quit` ou `Ctrl+D` encerram; `limpar`/`reset` apaga o
contexto da conversa.

Com `CITATION_LINK_SCHEME=vscode` (default), o arquivo na tabela de fontes e as
citações `[n]` abrem no VSCode com `Ctrl`/`Cmd` + clique. Use `file` para o app
padrão ou `none` para desativar.

## Configuração

Principais variáveis do `.env` (lista completa em `.env.example`):

| Variável | Default | Efeito |
|---|---|---|
| `OPENROUTER_API_KEY` | — | chave `sk-or-...` |
| `LLM_MODEL` | `openai/gpt-4o-mini` | modelo de chat |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | modelo de embedding |
| `FAISS_INDEX_DIR` | `./faiss_index` | diretório do índice |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | chunking |
| `RETRIEVAL_K` | `4` | trechos por pergunta |
| `FANOUT_ENABLED` / `FANOUT_MAX_QUERIES` | `true` / `3` | fan-out de consultas |
| `SUFFICIENCY_MAX_ROUNDS` | `1` | buscas extras se faltar contexto (`0` desliga) |
| `CHAT_MEMORY_TURNS` | `6` | turnos lembrados (`0` desliga) |
| `CITATION_LINK_SCHEME` | `vscode` | `vscode`, `file` ou `none` |
| `HYBRID_SEARCH` / `HYBRID_WEIGHTS` | `true` / `0.5,0.5` | busca híbrida |
| `TEMPERATURE` | `0` | temperatura do LLM |

Modelos no OpenRouter exigem prefixo de provider (`openai/...`); a aplicação
adiciona `openai/` e avisa se você omitir.

## Arquitetura

```text
ingest:  arquivos → loaders → chunks → embeddings → FAISS (disco)

chat:    pergunta → planner (1..N consultas) → busca híbrida + RRF
                 → suficiência (busca extra) → prompt → LLM → resposta + fontes
```

- `FAISS` guarda os vetores; `BM25` cobre termos exatos; RRF funde os rankings.
- O índice (`faiss_index/`) tem `index.faiss`, `index.pkl` e `manifest.json`.
- Explicação visual e componente por componente:
  [docs/ARQUITETURA.md](docs/ARQUITETURA.md).
- Plano de reranker e pool maior de candidatos:
  [plans/reranker-e-candidatos.md](plans/reranker-e-candidatos.md).

## Testes

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Os testes usam embeddings falsos: não consomem API nem exigem chave.

## Limitações

- Sem OCR: PDFs escaneados precisam ser convertidos antes.
- BM25 otimizado para inglês; outros idiomas exigem ajustar `tokenizer.py`.
- Sem reranker/gate: consultas sem sentido ainda recuperam vizinhos próximos.
- Memória não persiste entre sessões; BM25 é reconstruído a cada chat.
- Não suporta crawler de sites, HTML, DOCX, Confluence ou Notion.
