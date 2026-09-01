# Nomenclatura Oficial dos Modelos — Artigo Científico (THOR-PIML)

Este documento estabelece o padrão canônico e rigoroso de nomenclatura para todos os modelos, baselines e referências utilizados no texto, tabelas, figuras e logs do projeto **THOR-PIML** para submissão ao artigo.

---

## 1. Tabela Canônica de Nomenclatura

| Identificador no Código | Nome Oficial no Artigo / Figuras | Descrição Arquitetural / Papel Científico |
| :--- | :--- | :--- |
| `obs` / `CHIRPS` | **Observado (CHIRPS)** | Ground Truth observacional diário CHIRPS 5.5km (GT V3). |
| `EQM` | **EQM (2012)** | *Empirical Quantile Mapping* (Gudmundsson et al., 2012 / Scikit-Downscale). Baseline estatístico clássico. |
| `ResLSTM` | **ResLSTM (2018)** | *Residual Long Short-Term Memory* (Kratzert et al., 2018 / NeuralHydrology). Baseline recorrente puro. |
| `TCN` | **TCN (2018)** | *Temporal Convolutional Network* (Bai et al., 2018 / CMU LocusLab). Baseline convolucional 1D puro. |
| `THOR-V7` | **THOR-V7 (Híbrido Temporal)** | Modelo híbrido 1D com fusão dinâmica (*Gated Fusion*) LSTM + TCN. Estudo de ablação sem visão espacial. |
| `THOR-V8` | **THOR-V8 (PIML Espaço-Temporal)** | **Arquitetura Proposta Principal**: CNN 2D Sinótica + Fusão Híbrida LSTM-TCN + Barreira Física Clausius-Clapeyron. |

---

## 2. Regras de Padronização

1. **Proibição de Termos de Desenvolvimento**: Termos como *"Master"*, *"Espacial"*, *"v6_antigo"* ou variações isoladas estão terminantemente proibidos em tabelas, figuras, legendas e textos do artigo.
2. **Consistência em Figuras**: Todos os scripts de plotagem (`src/generate_paper_figures.py`, `src/plot_comparison.py`, etc.) devem consumir o dicionário canônico `COLOR_MAP` e as strings acima.
3. **Métricas Físicas**: Restrições termodinâmicas (ex: *Violação Física Clausius-Clapeyron*) aplicam-se apenas a modelos com barreira física (*Physics-Informed*). Baselines não-físicos (EQM, ResLSTM, TCN) devem ser marcados explicitamente como **`—` (Não aplicável)**.
