# THOR-PIML: Arquitetura Neural Híbrida com Física Informada para Modelagem e Downscaling Estatístico de Precipitação em Bacias Hidrográficas Regionais

[![Conference](https://img.shields.io/badge/CONIC-2026-blue.svg)](https://github.com/Zentsy/THOR-PIML-Conic2026)
[![License](https://img.shields.io/badge/License-Restricted%20Academic-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-ee4c2c.svg)](https://pytorch.org/)

> **Repositório Oficial do Artigo:**  
> *"THOR-PIML: Arquitetura Neural Híbrida com Física Informada para Modelagem e Downscaling Estatístico de Precipitação em Bacias Hidrográficas Regionais"*  
> **THOR-PIML:** *Taylor-Hurdle Optimized Regional Physics-Informed Machine Learning*  
> Submetido ao **CONIC 2026**.

---

## 📌 Material Suplementar

> [!IMPORTANT]
> **Aos Avaliadores do CONIC 2026 e Pesquisadores:**  
> O detalhamento matemático das formulações, derivações das funções de perda, deduções das métricas de avaliação e testes adicionais encontram-se descritos no **Material Suplementar Oficial** localizado no diretório:
> 📄 **[`docs/`](docs/)**

---

## 🔬 Arquitetura do Modelo (THOR-V8 PIML)

A arquitetura neural **THOR-V8** integra extração de padrões sinóticos 2D, dinâmica temporal multi-escala acoplada, decodificação Hurdle dual-head e barreira física termodinâmica:

<p align="center">
  <img src="results/figures/fig_thor_v8_architecture.png" alt="Arquitetura THOR-V8 PIML" width="95%"/>
</p>

1. **Entradas Pareadas (Lookback $T = 30$ dias):**
   - Campo Sinótico 2D (ERA5-PL): Tensor $(30, 25, 33, 5)$ com $z_{500}, u_{700}, v_{700}, q_{700}, w_{500}$.
   - Preditores de Superfície 1D (ERA5-Land): Tensor $(30, 84)$ contendo 16 variáveis locais + 80 lags temporais ($t-1$ a $t-14$).
2. **Encoder 2D:** Convoluções espaciais $(32 \to 64 \to 128) + \text{AvgPool} + \text{Linear} \to \mathbf{Z}_{\text{syn}} \in \mathbb{R}^{30 \times 64}$.
3. **Tronco Duplo ($30 \times 148$):** Concatenação $[\text{1D} \mathbin{\Vert} \mathbf{Z}_{\text{syn}}]$ alimentando em paralelo o **Ramo A** ($2\times \text{LSTM}(128) + \text{Residual}$) e o **Ramo B** ($\text{Conv1D Causal}$ multi-escala $k \in \{3,5,7\}, d \in \{1,2,4,8\}$).
4. **Fusão & Atenção:** Fusão Gated adaptativa $\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{h}_{\text{lstm}} + (1 - \mathbf{g}) \odot \mathbf{h}_{\text{tcn}}$ seguida de Atenção Causal com 8 cabeças ($\text{SDPA}$) fatiando o estado $\mathbf{h}_T \in \mathbb{R}^{128}$.
5. **Saída Hurdle:** Decodificador de Ocorrência ($\text{Sigmoid} \to p_{\text{occ}} \in [0, 1]$) $\times$ Decodificador de Intensidade ($\text{Softplus} \to \mu_{\text{int}} \ge 0$), gerando $\hat{y} = p_{\text{occ}} \times \mu_{\text{int}}$ (mm/dia).
6. **Barreira Física:** Restrição termodinâmica de Clausius-Clapeyron $\mathcal{L}_{\text{phys}} = \lambda \cdot \text{Softplus}(\hat{y} - 4.0 \cdot \text{TCWV})^2$, garantindo **0.00% de violação física**.

---

## 📊 Benchmark Oficial (Teste Cego: 2019–2026, $N = 2.494$ dias)

| Modelo | Paradigma | KGE (↑) | R10mm (Dias $\ge$ 10mm) | R20mm (Tempestades $\ge$ 20mm) | Violação Clausius-Clapeyron |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EQM (Gudmundsson 2012)** | Estatístico Empírico | $-0.005$ | 77 / 287 | 10 / 106 | 0.00% |
| **ResLSTM (Kratzert 2018)** | Recorrente com Residual | $+0.264$ | 241 / 287 | 81 / 106 | 0.00% |
| **TCN (Bai 2018)** | Convolucional Temporal | $+0.248$ | 240 / 287 | 79 / 106 | 0.00% |
| **THOR-V7 (Proposta)** | Híbrido Temporal (LSTM+TCN) | $+0.395$ | 272 / 287 | 93 / 106 | 0.00% |
| **THOR-V8 PIML (Proposta)** | **2D Synoptic CNN + Híbrido PIML** | **+0.410** | **279 / 287** | **98 / 106** | **0.00%** |

*Observado no período de teste cego: 287 dias com chuva $\ge 10\text{ mm}$ e 106 dias com tempestades $\ge 20\text{ mm}$.*

---

## 🚀 Reprodução em 1 Comando

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar benchmark e gerar as figuras oficiais
python reproduce_paper_results.py
```

Os checkpoints pré-treinados estão disponíveis em `checkpoints/` e os resultados numéricos são consolidados em `results/`.

---

## 📂 Estrutura do Repositório

```text
THOR-PIML-Conic2026/
├── checkpoints/              # Checkpoints pré-treinados (.pt) e scaler
│   ├── v8_hybrid_seed42.pt   # Modelo campeão THOR-V8 PIML
│   ├── v7_hybrid_v7_v3_seed42.pt
│   ├── v7_lstm_v7_v3_seed42.pt
│   └── v7_tcn_v7_v3_seed42.pt
├── data/                     # Dataset tabular de treino e teste cego
│   └── ground_truth_guarulhos_daily_v3.csv
├── docs/                     # Material Suplementar
├── results/                  # Tabela do benchmark e figuras
│   └── figures/              # Figuras oficiais do artigo (PNG)
├── src/                      # Implementação das redes neurais e perdas físicas
├── reproduce_paper_results.py # Script de reprodução automática
└── requirements.txt          # Dependências do projeto
```

---

## 📜 Licença

Uso estritamente acadêmico sob a **[Restricted Academic Evaluation & Research License](LICENSE)**. Proibido qualquer uso comercial ou governamental/estatal sem autorização formal prévia dos autores.
