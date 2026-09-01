# THOR-PIML V7/V8 — Arquitetura Híbrida LSTM+CNN (branch `hybrid-arch-test`)

> **Status:** em desenvolvimento (2026-08-15). Este é o documento vivo da era híbrida.
> Histórico fechado V1→V6d: `ESTADO_ATUAL_FINAL_V6.md` (não duplicar aqui).
> Bibliografia anotada para o paper: `PAPER_REFERENCES.md`.

## 1. Diagnóstico que motiva a híbrida (herdado do V6d)

O V6d bateu no **teto informacional + arquitetural** com ResBiLSTM puro sobre 30 dias
de preditores de superfície (T/RH/PS de 1 ponto):

| Falha V6d | Causa raiz | Ataque na V7/V8 |
|---|---|---|
| NSE −0.18 | sem informação sinótica de alto nível | V8: ERA5 PL (z500, u/v/q/w 700/500) via CNN espacial |
| R20 recall 13% | loss pune pouco FN de extremos | V7: termo extremes-recall (§4.6) |
| R10 2× over | occurrence gate vaza garoa | V7: TCN multi-escala + fusão gated |
| CWD 210 vs 29 | garoa persistente (drizzle bias) | V7: variance matching + CV por KGE |
| QB99 −36% | quantile 0.65 conservador | V7: extremes-recall moderado; V8: CAPE/TCWV reais |

## 2. Roadmap em fases

| Fase | Entrega | Estado |
|---|---|---|
| **V7 temporal** | Híbrido LSTM+TCN com dados atuais (84 feats reais), loss V7, OneCycle, CV bloqueada | ✅ código (smoke CPU ok) |
| **Dados novos** | CHIRPS grade 30 células (.nc locais), ERA5 PL domínio, CAPE/TCWV reais, GT V3 | ✅ scripts prontos, rodar na lightning.ai |
| **V8 espacial** | CNN 2D sobre domínio sinótico ERA5 + física CC com TCWV real | ✅ código (smoke sintético ok) |
| **Paper** | Ablação completa + protocolo de validação | `PAPER_REFERENCES.md` + `results/experiments_v7.md` |

## 3. Arquitetura V7 (`src/v7/model_v7.py`)

```
x (B, 30, F)  F=84 (V2) | F=84+9 sinóticas/espaciais (GT V3)
├── Ramo LSTM: ResBiLSTM 128×3, dropout 0.3, causal (memória/inércia)
├── Ramo TCN:  4 blocos multi-escala, kernels {3,5,7} paralelos,
│              dilations {1,2,4,8} (campo receptivo 91d), GELU, residual, causal
└── Fusão:     g = σ(W[lstm;tcn]) por timestep → g·lstm + (1−g)·tcn
    → SDPA 8 heads causal (residual) → LayerNorm → último passo
    → heads hurdle: occurrence(sigmoid) × intensity(softplus)
```

- **Por que os dois ramos:** LSTM captura persistência (garoa, inércia de 14d);
  TCN detecta formas de padrão (frente fria 2-4d, onda 7-14d) com gradiente estável
  e paralelismo (Bai et al. 2018). A fusão gated aprende **quando** confiar em cada ramo.
- **Causalidade total:** LSTM unidirecional, convs causais (pad esquerda), máscara
  causal na SDPA (lição V3).
- **Ablação sem código novo:** `--model lstm | tcn | hybrid` (switches na config).
- ~1.42M params (T4 tranquila; dataset ~11.6k janelas → dropout 0.3 + wd 1e-3 + CV).

## 4. Loss V7 (`src/v7/physics_loss_v7.py`)

Base = V6d balanceada (q=0.65, storm 1.3×/2.0×, underest 1.3×/1.5×, dry 0.3,
final 0.2, sharpness 0.05) + dois termos novos:

6. **Extremes-recall:** `λ·mean(FN_R10·1.5·relu(y−ŷ) + FN_R20·2.5·relu(y−ŷ))`, λ=0.5.
   Docs §9 pedia ×10; moderamos pela lição V6c (6× total virou R10 1252).
7. **Variance matching simétrico:** `0.02·|std_pred − std_true|` (o sharpness V6d
   era one-sided e permitia estourar para cima).

Física CC softplus permanece **λ=0 no V7** (tcwv V2 é constante 20mm — barrier é
ruído). **V8 reativa** com TCWV real do ERA5 single-levels.

## 5. Protocolo de validação (muda vs V6 — importante para o paper)

1. **CV temporal bloqueada** (expanding window, 5 blocos) **só na região dev**
   (primeiros 85%, 1981→2019-09). `pipeline_v7.blocked_temporal_folds`.
2. **Teste cego** 2019-09-02→2026-06-30: intocado até o relatório final de cada
   variante. Comparável com todos os números V5/V6 históricos (mesmo split).
3. **Seleção de época por val KGE** (a config V6 dizia "kge" mas o train.py
   ignorava — no V7 é real). Early stop na val loss.
4. **OneCycleLR** 3e-4→1e-3→3e-5 (docs §9), grad clip 1.0, AMP T4.
5. Toda corrida appenda em `results/experiments_v7.md` (rastreabilidade do paper).

## 6. Bugs corrigidos na era V7 (checar se regressar)

- **Lag-14 fantasma:** `preprocessing.py` chamava `engineer_temporal_lags` sem os
  lags V6 → "84 feats" treinava com 70 desde sempre. Corrigido; `run_evaluation`
  recria o set antigo (70) via `infer_lags_from_feature_names` para checkpoints
  históricos continuarem avaliáveis.
- **run_evaluation hardcoded V5:** avaliava checkpoint V6 com classe V5. Agora
  `detect_thor_version` + construção correta (V5/V6/V7) a partir do checkpoint.

## 7. Dados novos (Phase 1 — scripts em `data_prep/`, rodar na lightning.ai)

| Script | Saída | Fonte |
|---|---|---|
| `extract_chirps_grid.py` | grade 5×6=30 células + stats diários (max/std/wet_frac) | `.nc` CHIRPS locais |
| `fetch_era5_pl.py` | z500, u/v/q/w 700/500 média+std do domínio (6°×8°, 0.25°) | CDS ERA5 PL |
| `fetch_era5_single.py` | CAPE real (max/média diária), TCWV real (média diária) | CDS single-levels |
| `build_gt_v3.py` | `ground_truth_guarulhos_daily_v3.csv` (cape/tcwv reais substituem proxy/constante) | merge |

Regras: anti-mock hard (`THOR_ALLOW_MOCK=1` explícito), provenance em toda saída,
provenance/regressão checada contra as séries V2 existentes.

## 8. V8 (espacial) — código pronto (`src/v7/model_v8.py`, `pipeline_v8.py`, `run_v8.py`, `physics_loss_v8.py`)

- `THORSpatialHybridModel`: `SpatialSynopticEncoder` (CNN 2D por dia sobre
  z500/u700/v700/q700/w500 do domínio → embedding 64d/dia) concatenado por timestep
  às features de superfície → tronco híbrido V7 (LSTM||TCN → gated → SDPA → hurdle).
  ~1.83M params. Fallback de ablação: rodar V7 `--data v3` (sinótica como features
  tabulares de domínio) compara "CNN aprende gradientes" vs "médias manuais".
- Barreira física CC REATIVADA (`THORLossV8`): softplus(ŷ − k·TCWV_real)², k=4
  (fator de reciclagem por convergência de umidade), λ=0.05 default.
- Janelamento validado: pares (surface `(N,30,F)`, spatial `(N,30,25,33,5)`) com
  alinhamento janela↔alvo testado (synthetic alignment test).
- Requer GT V3 + `data/era5pl_domain_daily_1981_2026.nc` (fase 1 na lightning.ai).
- Stretch (ainda não implementado): head multi-tarefa prevendo as 30 células CHIRPS.

## 9. Como rodar

```bash
# T4 (lightning.ai) — linha principal:
python -m src.v7.run --model hybrid --loss v7 --data v2 --cv 5 --seed 42
# Ablações: --model lstm | tcn ; --loss v6d
# Smoke local CPU:
python -m src.v7.run --smoke
# V8 espacial (após fase 1 + GT V3):
python -m src.v7.run --model hybrid --data v3 --cv 5   # sinótica como features tabulares
python -m src.v7.run_v8 --model hybrid --physics 0.05  # CNN 2D sobre o domínio
python -m src.v7.run_v8 --smoke-synthetic              # valida shapes sem dados
```
