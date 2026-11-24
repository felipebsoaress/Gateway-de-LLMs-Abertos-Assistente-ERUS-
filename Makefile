.PHONY: help setup db-start ollama-start ingest run stop

help:
	@echo "🤖 Gateway ERUS - Comandos disponíveis:"
	@echo "  make setup    - Instala dependências (Python) e baixa os modelos (Ollama)"
	@echo "  make start-bg - Sobe o Qdrant (Docker) e os 2 nós do Ollama em background"
	@echo "  make ingest   - Lê os PDFs e o site oficial para popular o Qdrant"
	@echo "  make run      - Inicia o Gateway (FastAPI) no terminal atual"
	@echo "  make stop     - Derruba o Qdrant e mata os processos do Ollama em background"

setup:
	pip3 install -r requirements.txt
	ollama pull qwen2.5-coder:1.5b
	ollama pull nomic-embed-text

start-bg:
	@echo "🧹 Limpando containers Qdrant antigos (se existirem)..."
	-docker stop qdrant_erus 2>/dev/null || true
	-docker rm qdrant_erus 2>/dev/null || true
	@echo "🚀 Subindo banco Qdrant..."
	docker run -d --name qdrant_erus -p 6333:6333 -p 6334:6334 qdrant/qdrant
	@echo "🧠 Iniciando nós do Ollama em background..."
	@if ! ss -tln | grep -q ":11434 "; then \
		OLLAMA_HOST=127.0.0.1:11434 nohup ollama serve > ollama_no1.log 2>&1 & \
	else \
		echo "ℹ️ Porta 11434 já está ativa (usando instância existente)."; \
	fi
	@if ! ss -tln | grep -q ":11435 "; then \
		OLLAMA_HOST=127.0.0.1:11435 nohup ollama serve > ollama_no2.log 2>&1 & \
	else \
		echo "ℹ️ Porta 11435 já está ativa."; \
	fi
	@echo "⏳ Aguardando nós estabilizarem..."
	sleep 3
	@echo "📦 Garantindo modelo no Nó 2 (11435)..."
	OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen2.5-coder:1.5b
	@echo "✅ Serviços prontos!"

ingest:
	@echo "📚 Iniciando ingestão de dados no Qdrant..."
	python3 ingest_site.py
	python3 ingest_redes.py

run:
	@echo "🌐 Iniciando Gateway..."
	python3 gateway_erus.py

stop:
	@echo "🛑 Parando o banco Qdrant..."
	-docker stop qdrant_erus
	-docker rm qdrant_erus
	@echo "🛑 Matando processos do Ollama..."
	-sudo pkill -f "ollama serve"
	@echo "🧹 Limpeza concluída!"