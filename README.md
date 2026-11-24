# Gateway de LLMs Abertos - Assistente ERUS

Este repositório contém a implementação do **Projeto da disciplina de Redes de Computadores (UFES)**, estendido para servir como um assistente de Inteligência Artificial focado na **Equipe de Robótica da UFES (ERUS)**. 

## 🎯 Intuito do Projeto
O objetivo central é aplicar conceitos práticos de redes (filas, roteamento, balanceamento, streaming e cache) no contexto de serviços de Inteligência Artificial local. O sistema fornece respostas sobre o material da disciplina de Redes e sobre as informações da equipe ERUS, rodando modelos de pesos abertos de forma independente, garantindo privacidade, baixa latência e total soberania dos dados.

## ⚙️ Como Funciona
A arquitetura é baseada em RAG (Retrieval-Augmented Generation) com um Gateway central responsável pelo roteamento e controle do tráfego.

* **Ingestão de Conhecimento:** Os scripts `ingest_redes.py` e `ingest_site.py` extraem informações (PDFs de slides e varredura de URLs do site oficial da ERUS), fragmentam os textos e usam o modelo `nomic-embed-text` para convertê-los em embeddings. Esses vetores são armazenados em um banco de dados vetorial Qdrant.
* **Roteamento por Domínio:** O servidor principal (`gateway_erus.py`, rodando em FastAPI na porta 8080) recebe perguntas e verifica o domínio solicitado. As requisições são então direcionadas ao nó Ollama apropriado com instruções de sistema específicas:
  * `erus-normas`: Roteia para a coleção `site_erus` usando o nó na porta 11434.
  * `erus-redes`: Roteia para a coleção `material_redes` usando o nó na porta 11435.
* **Cache Semântico:** Antes de repassar uma pergunta ao gerador, o sistema verifica no Qdrant se uma pergunta com significado semântico (distância por cosseno) muito semelhante (limiar > 0.95) já foi respondida recentemente. Se sim, ele devolve imediatamente o conteúdo cacheado, economizando GPU e atingindo TTFT (Time To First Token) quase zero.
* **Geração Distribuída e Streaming:** O conteúdo recuperado é injetado no prompt do modelo `qwen2.5-coder:1.5b`. A resposta é devolvida ao cliente de forma contínua, token a token, enquanto métricas de rede (latência total, TTFT e Tokens/s) são monitoradas em tempo real.

---

## 🚀 Como Fazer Rodar

### Pré-requisitos
* Python 3.x
* [Docker](https://www.docker.com/) (para rodar o banco vetorial Qdrant)
* [Ollama](https://ollama.com/) (para os modelos locais)
* PDFs de estudo da disciplina devem estar dentro da pasta `slides_redes/` na raiz do projeto.

### Passos de Execução
Todo o fluxo de execução é gerenciado pelo arquivo `Makefile` fornecido.

**1. Instalação de dependências e download dos modelos**
```bash
make setup
```
*Isso instalará as bibliotecas Python (FastAPI, Qdrant, BeautifulSoup, etc.) e fará o pull dos modelos `qwen2.5-coder:1.5b` e `nomic-embed-text`.*

**2. Subir os serviços em background (Banco de Dados e Nós Ollama)**
```bash
make start-bg
```
*Inicia o container Docker do Qdrant (portas 6333 e 6334) e levanta dois nós independentes do Ollama simulando o ambiente distribuído.*

**3. Popular o Banco Vetorial (Ingestão RAG)**
```bash
make ingest
```
*Extrai e converte todo o texto do site da ERUS e dos PDFs em vetores.*

**4. Iniciar o Gateway Principal**
```bash
make run
```
*Inicia o servidor FastAPI. O assistente ficará disponível e ouvindo requisições na porta 8080.*

### Interrompendo a Execução
Para limpar os processos locais do Ollama e parar o container do banco de dados, execute:
```bash
make stop
```
