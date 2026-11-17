# RAG Híbrido (PoC)

PoC de um sistema RAG híbrido via CLI: indexa PDFs, TXTs e Markdown/MDX em um
índice **FAISS** local, combina busca **vetorial** com busca **lexical (BM25)**
e responde perguntas com um LLM, sempre citando as fontes. Embeddings e LLM são
acessados com **uma única chave do OpenRouter**.

A indexação é otimizada para **inglês** (stopwords + stemming Snowball),
adequada para documentações técnicas como a do React. As respostas continuam
em português.

## Recursos

- `ingest`: indexa um arquivo (`.pdf`, `.txt`, `.md`, `.mdx`, `.markdown`) ou
  diretório, com chunking (`chunk_size=1000`, `chunk_overlap=200`),
  deduplicação por hash e persistência em `./faiss_index`.
- `ingest --dry-run`: estima chunks, tokens, custo de embedding, custo por
  pergunta e tempo, **sem chamar a API**.
- `chat`: loop interativo com streaming token a token, memória dos últimos
  turnos, reformulação de follow-ups para busca, prompt que prioriza o contexto
  mas permite complemento com conhecimento geral (sinalizado e sem citação),
  citação `[n]` apenas para fontes do contexto e tabela de fontes clicável.
- `clear`: remove o índice local.
- `status`: mostra o manifesto do índice e a configuração atual.
- Busca híbrida configurável via `HYBRID_SEARCH` (FAISS + BM25, fundidos por
  Reciprocal Rank Fusion).

## Requisitos

- Python 3.10+
- Chave do [OpenRouter](https://openrouter.ai/keys): usada para
  `openai/gpt-4o-mini` (chat) e `openai/text-embedding-3-small` (embeddings).

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# edite .env e preencha OPENROUTER_API_KEY
```

Opcional (habilita o comando `rag` no PATH e permite `python -m rag_poc`):

```bash
pip install -e .
```

## Uso

```bash
# 0) estimar custo/tempo antes de gastar (não chama a API)
rag ingest /caminho/para/react.dev/src/content --dry-run

# 1) indexar um diretório (recursivo) ou um arquivo
rag ingest ./docs
rag ingest ./docs/contrato.pdf

# reindexar do zero
rag ingest ./docs --rebuild

# 2) conversar com os documentos
rag chat
rag chat --k 6 --show-chunks

# 3) inspecionar / limpar
rag status
rag clear            # pede confirmação
rag clear --yes
```

Sem `pip install -e .`, use o módulo:

```bash
PYTHONPATH=src python -m rag_poc ingest ./docs
PYTHONPATH=src python -m rag_poc chat
```

Dentro do `chat`, digite `sair`, `exit`, `quit` ou `Ctrl+D` para encerrar, e
`limpar` (ou `reset`) para apagar o contexto da conversa.

### Links nas fontes

Por padrão (`CITATION_LINK_SCHEME=vscode`), o nome do arquivo na tabela de
fontes vira um hyperlink OSC 8: `Ctrl`/`Cmd` + clique abre no VSCode. Opções:

- `vscode`: `vscode://file/<caminho absoluto>`;
- `file`: `file:///<caminho>` (app padrão; PDFs usam `#page=N`);
- `none`: desativa.

O link usa o caminho absoluto do `source` no índice. Arquivos ingeridos por
caminho relativo são resolvidos a partir do diretório atual; prefira caminhos
absolutos no `ingest`. Ainda não há link para a linha exata do chunk (o
`start_index` do Markdown é relativo ao texto já limpo).

### Exemplo de sessão

```text
$ rag ingest ./docs
2 arquivo(s) encontrado(s) em 'docs'.
        Ingestão concluída
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Modelo de embedding      ┃ openai/text-embedding-3-small ┃
┃ Dimensões                ┃ 1536                    ┃
┃ Chunks gerados           ┃ 42                      ┃
┃ Chunks novos             ┃ 42                      ┃
┃ Vetores no índice        ┃ 42                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛

$ rag chat
╭──────────── RAG Híbrido ────────────╮
│ Modelo: openai/gpt-4o-mini          │
│ Busca: híbrida (FAISS + BM25) · k=4 │
╰─────────────────────────────────────╯

Você › Qual é o prazo de rescisão?
Resposta (4 trecho(s) recuperado(s))
O prazo é de 30 dias corridos [1].

         Fontes
┏━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ # ┃ Arquivo   ┃ Página ┃ Via         ┃ Citada ┃
┡━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ 1 │ contrato.pdf │ 3   │ vetor+bm25  │ sim    │
│ 2 │ anexo.pdf    │ 1   │ vetor       │ —      │
└───┴───────────┴────────┴─────────────┴────────┘
```

## Estimativa de custo (`--dry-run`)

O `--dry-run` lê e faz chunk de tudo localmente, conta os tokens com o encoding
`cl100k_base` (o mesmo do `text-embedding-3-small`) e imprime:

- nº de chunks e de tokens de embedding;
- custo estimado de embedding (tabela de preços em `estimation.py`);
- faixa de tempo das chamadas de embedding (lotes de `EMBEDDING_BATCH_SIZE`);
- tempo de processamento local medido;
- custo aproximado por pergunta no chat (contexto de `k` chunks + overhead).

Nada é enviado à API e o índice não é alterado. Os preços usados são de
referência (USD por 1M tokens) e podem ser ajustados em `estimation.py`.

## Configuração (`.env`)

| Variável | Default | Descrição |
|---|---|---|
| `OPENROUTER_API_KEY` | — | chave `sk-or-...` (obrigatória em `ingest`/`chat`) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | endpoint OpenAI-compatível |
| `LLM_MODEL` | `openai/gpt-4o-mini` | modelo de chat |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | modelo de embedding |
| `EMBEDDING_DIMENSIONS` | vazio (default do modelo) | dimensão reduzida; exige reindexar |
| `EMBEDDING_BATCH_SIZE` | `64` | textos por requisição de embedding |
| `TEMPERATURE` | `0` | temperatura do LLM |
| `FAISS_INDEX_DIR` | `./faiss_index` | diretório do índice |
| `CHUNK_SIZE` | `1000` | tamanho do chunk |
| `CHUNK_OVERLAP` | `200` | sobreposição entre chunks |
| `RETRIEVAL_K` | `4` | trechos recuperados por pergunta |
| `CHAT_MEMORY_TURNS` | `6` | turnos lembrados no chat; `0` desativa a memória |
| `CITATION_LINK_SCHEME` | `vscode` | links nas fontes: `vscode`, `file` ou `none` |
| `HYBRID_SEARCH` | `true` | liga/desliga o BM25 |
| `HYBRID_WEIGHTS` | `0.5,0.5` | pesos vetorial,BM25 (normalizados) |
| `APP_URL` / `APP_NAME` | `http://localhost` / `rag-poc` | headers de atribuição do OpenRouter |

Modelos no OpenRouter exigem **prefixo de provider** (`openai/...`). Se você
omitir, a aplicação adiciona `openai/` e avisa.

Com `TEMPERATURE=0` (default), a mesma pergunta + mesmo contexto tendem a gerar
a mesma resposta. A flexibilidade é por conteúdo (o modelo explica *o que*
faltou); variação estilística entre execuções exige temperatura maior.

## Arquitetura

Para uma explicação visual, componente por componente e passo a passo do fluxo de dados, veja [docs/ARQUITETURA.md](docs/ARQUITETURA.md).

```text
ingest:  arquivos ──► loaders ──► chunk_documents ──► embeddings (OpenRouter)
                                                        │
                                                        ▼
                                            FAISS (index.faiss/index.pkl)
                                                        │
chat:    pergunta ──► Retriever ──► prompt LCEL ──► LLM (streaming) ──► resposta
                        │  │
                        │  └─ BM25 (rank_bm25, tokenização PT-BR)
                        └─ FAISS (similaridade)
                        └─ fusão RRF
```

- `config.py`: `.env` → `Settings` validado (chave, modelos, chunking, pesos).
- `loaders.py`: descoberta recursiva, PDF por página, TXT com fallback de encoding.
- `loaders.py`: Markdown tem frontmatter, imports MDX e comentários removidos;
  blocos de código são preservados.
- `indexer.py`: chunking, `doc_id` determinístico (SHA-256 de
  `source|page|start_index`), dedupe no reingest, manifest com modelo/dimensão.
- `tokenizer.py`: normalização para inglês, stopwords e stemming Snowball
  (`components`/`component` -> `compon`).
- `estimation.py`: contagem de tokens com `tiktoken` e preços de referência.
- `retriever.py`: FAISS + BM25 com RRF; `sources` indica `vetor`, `bm25` ou ambos.
- `chain.py`: prompt que usa o contexto como fonte primária, permite
  complemento com conhecimento geral explicitamente sinalizado e proíbe citações
  fora do contexto; cadeia LCEL `prompt | llm | StrOutputParser()` e chain de
  reformulação de consulta (follow-up -> pergunta autônoma).
- `memory.py`: histórico em memória limitado a `CHAT_MEMORY_TURNS`, com `limpar`.
- `ui.py`: streaming em Markdown, tabelas e mensagens de erro.
- `cli.py`: Typer; traduz erros da API em mensagens acionáveis.

### Por que híbrido?

Busca vetorial captura paráfrases e semântica, mas pode perder termos exatos
(códigos, siglas, números). BM25 cobre esses casos. A fusão por RRF combina os
rankings sem precisar comparar scores de naturezas diferentes — e cada fonte
recuperada informa por qual caminho veio.

## Tratamento de erros

- Sem `OPENROUTER_API_KEY`: mensagem com instrução para criar o `.env`.
- Caminho inexistente, extensão não suportada, PDF escaneado/vazio: erro por
  arquivo, sem abortar o lote; resumo final lista as falhas.
- Índice ausente: pede para rodar `rag ingest`.
- Índice criado com outro modelo/dimensão: pede `rag clear` e reindexação.
- `401` chave inválida, `402` sem créditos, `404` modelo inexistente, `429`
  rate limit: mensagens específicas; `--debug` mostra o traceback.
- `Ctrl+C`/`Ctrl+D` no chat encerram sem stacktrace.

## Desenvolvimento

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

Os testes usam embeddings falsos: não consomem API nem exigem chave.

## Limitações da PoC

- Não há OCR: PDFs escaneados precisam ser convertidos para texto antes.
- A busca lexical é otimizada para inglês; para outros idiomas, troque a lista
  de stopwords e o stemmer em `tokenizer.py`.
- BM25 é reconstruído em memória a cada sessão de chat (adequado para PoCs;
  para grandes corpora vale persistir o índice lexical).
- A memória é em processo: limitada a `CHAT_MEMORY_TURNS` e perdida ao sair.
  Cada turno com histórico faz uma chamada extra ao LLM para reformular a
  consulta de busca (custo adicional desprezível).
- `faiss_index/` é gerado localmente e não deve ser versionado.
