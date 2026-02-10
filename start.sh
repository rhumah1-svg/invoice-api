#!/bin/bash
# ============================================
# Invoice Extraction API — Lancement tout-en-un
# ============================================
# Usage: ./start.sh
# ============================================

set -e

# ── 1. Vérifier Python ──
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé. Installe-le via https://python.org"
    exit 1
fi

echo "✅ Python trouvé: $(python3 --version)"

# ── 2. Charger ou demander la clé OpenAI ──
if [ -f ".env" ]; then
    source .env
    export OPENAI_API_KEY
    echo "✅ Clé OpenAI chargée depuis .env"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    read -p "🔑 Entre ta clé OpenAI (sk-...): " OPENAI_API_KEY
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
    echo "✅ Clé sauvegardée dans .env (tu n'auras plus à la retaper)"
    export OPENAI_API_KEY
fi

# ── 3. Créer un environnement virtuel ──
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

source venv/bin/activate

# ── 4. Installer les dépendances ──
echo "📥 Installation des dépendances..."
pip install --quiet fastapi==0.115.6 uvicorn[standard]==0.34.0 pdfplumber==0.11.4 openai==1.58.1 pydantic==2.10.3 python-multipart==0.0.19

# ── 5. Lancer le serveur ──
echo ""
echo "🚀 Serveur lancé !"
echo "   → Interface web : http://localhost:8000/docs"
echo "   → Endpoint API  : http://localhost:8000/extract"
echo "   → Ctrl+C pour arrêter"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
