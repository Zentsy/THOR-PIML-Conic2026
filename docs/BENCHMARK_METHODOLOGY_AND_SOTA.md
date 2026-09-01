# THOR-PIML — Metodologia de Benchmark e Estado da Arte (SOTA)
===================================================================

> **Documento de Auditoria e Referência Científica.**  
> Todas as referências externas possuem **links diretos (DOIs)** clicáveis para os artigos originais publicados.  
> As métricas do baseline histórico são dados **reais medidos no repositório**. As estimativas de V7/V8 são tratadas como **hipóteses experimentais a serem validadas**.

---

## 1. As 4 Arenas de Avaliação do Benchmark

Na literatura de downscaling climático e hidrologia computacional, a avaliação de modelos de precipitação diária é dividida em 4 arenas padronizadas:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ARENAS DO BENCHMARK THOR                                  │
├──────────────────────┬──────────────────────┬───────────────────┬──────────────────────┤
│ 1. Precisão Geral    │ 2. Extremos Críticos │ 3. Regime de Seca │ 4. Física e Limites  │
│    (Hidrologia)      │    (WMO ETCCDI)      │    (Ocorrência)   │    (Termodinâmica)   │
├──────────────────────┼──────────────────────┼───────────────────┼──────────────────────┤
│ • KGE (Kling-Gupta)  │ • R20 Recall (%)     │ • F1-Score Chuva  │ • Violação CC (%)    │
│ • NSE (Nash-Sutcl.)  │ • QB99% Bias (%)     │ • Brier Score     │ • W_max Realista     │
│ • RMSE & MAE (mm)    │ • QB95% Bias (%)     │ • CWD (Dias Chuv.)│ • Conservação Massa  │
│ • Bias Médio (mm)    │ • SDII (Intensidade) │ • CDD (Dias Secos)│                      │
└──────────────────────┴──────────────────────┴───────────────────┴──────────────────────┘
```

---

## 2. Métricas Oficiais e Suas Definições Científicas

### Arena 1: Precisão Hidrológica Geral
* **KGE (Kling-Gupta Efficiency)**: Métrica padrão-ouro recomendada para hidrologia e clima ([Gupta et al., 2009](https://doi.org/10.1016/j.jhydrol.2009.08.003); [Kling et al., 2012](https://doi.org/10.1016/j.jhydrol.2012.01.011)). Decompõe o erro em 3 componentes balanceados:
  $$KGE = 1 - \sqrt{(r - 1)^2 + (\beta - 1)^2 + (\gamma - 1)^2}$$
  onde $r$ é a correlação de Pearson, $\beta = \mu_{\text{pred}}/\mu_{\text{obs}}$ é o viés relativo, e $\gamma = \sigma_{\text{pred}}/\sigma_{\text{obs}}$ é a razão de variabilidade. Range: $(-\infty, 1.0]$.
* **NSE (Nash-Sutcliffe Efficiency)**: Métrica clássica da hidrologia ([Nash & Sutcliffe, 1970](https://doi.org/10.1016/0022-1694(70)90255-6)). Range: $(-\infty, 1.0]$.

### Arena 2: Extremos Climáticos (Convenção WMO / ETCCDI)
Definidos pelo *Expert Team on Climate Change Detection and Indices* ([Zhang et al., 2011](https://doi.org/10.1002/wcc.147)):
* **R10mm / R20mm**: Contagem de dias no período com precipitação diária $\ge 10\text{ mm}$ (chuva forte) e $\ge 20\text{ mm}$ (chuva muito forte / risco de enchente).
* **R20 Recall (%)**: Capacidade do modelo de detectar dias de chuva $\ge 20\text{ mm}$ observados: $\frac{\text{Preditos } \ge 20\text{ mm em dias reais } \ge 20\text{ mm}}{\text{Total de dias observados } \ge 20\text{ mm}} \times 100$.
* **QB95 / QB99 (Quantile Bias)**: Viés relativo nos percentis 95 e 99 da distribuição de chuva: $\frac{Q_{99}^{\text{pred}} - Q_{99}^{\text{obs}}}{Q_{99}^{\text{obs}}} \times 100$.
* **SDII (Simple Daily Intensity Index)**: Precipitação média acumulada apenas nos dias chuvosos ($\ge 1.0\text{ mm}$): $\frac{\sum P_{\text{wet}}}{N_{\text{wet}}}$.

### Arena 3: Dinâmica de Seca e Ocorrência
* **CWD (Consecutive Wet Days)**: Maior número de dias consecutivos com chuva $\ge 1.0\text{ mm}$. Avalia se o modelo sofre do vício de "garoa infinita".
* **CDD (Consecutive Dry Days)**: Maior sequência de dias secos consecutivos ($< 1.0\text{ mm}$).
* **F1-Score / Accuracy de Ocorrência**: Avaliação da cabeça hurdle de classificação binária (se choveu ou não).
* **Brier Score**: Calibração probabilística da probabilidade prevista vs ocorrência real ([Brier, 1950](https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2)).

### Arena 4: Consistência Física (PIML)
* **Violação da Barreira de Clausius-Clapeyron**: Porcentagem de dias onde a chuva prevista $\hat{y}$ ultrapassa o limite termodinâmico de água precipitável disponível na coluna: $\hat{y} > 4 \times TCWV_{\text{real}}$ ([Allen & Ingram, 2002](https://doi.org/10.1038/nature01092); [Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045)).

---

## 3. Literatura Científica e Métodos de Comparação (com DOIs)

| Método / Modelo | Artigo Original | DOI Oficial | Papel no Benchmark |
| :--- | :--- | :--- | :--- |
| **DeepESD (CNN)** | Baño-Medina et al. (2020), *Geosci. Model Dev.* | [10.5194/gmd-13-2109-2020](https://doi.org/10.5194/gmd-13-2109-2020) | Padrão Deep Learning CNN do CORDEX |
| **Quantile Delta Mapping (QDM)** | Cannon et al. (2015), *Journal of Climate* | [10.1175/JCLI-D-14-00754.1](https://doi.org/10.1175/JCLI-D-14-00754.1) | Padrão-ouro estatístico de correção de viés |
| **DeepSD (Super-Resolution)** | Vandal et al. (2017), *ACM SIGKDD* | [10.1145/3097983.3098004](https://doi.org/10.1145/3097983.3098004) | Abordagem clássica de super-resolução para clima |
| **Hurdle-DL para Extremos** | DEMM (2022) / Hurdle-IMDL (2025) | [KDD DEMM PDF](https://www.cse.msu.edu/~ptan/papers/kdd2022.pdf) · [arXiv:2510.20486](https://arxiv.org/html/2510.20486v1) | Estado da arte de redes neurais com hurdle |
| **Física-Informada (PINNs)** | Raissi, Perdikaris & Karniadakis (2019), *JCP* | [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) | Fundamento do aprendizado informado pela física |
| **Clausius-Clapeyron Scaling** | Allen & Ingram (2002), *Nature* | [10.1038/nature01092](https://doi.org/10.1038/nature01092) | Limite físico da taxa de precipitação vs vapor |

---

## 4. O Baseline Histórico Oficial do Repositório (Fato Medido)

Os números abaixo são os dados **reais, verificáveis e congelados** obtidos pelo modelo **THOR V6d** no conjunto de teste cego (2019-09 a 2026-06), arquivados em `docs/archive/ESTADO_ATUAL_FINAL_V6.md` (commit `bd86455`):

```json
{
  "modelo": "THOR V6d (ResBiLSTM 128x3, 84 feats, Loss V6d)",
  "periodo_teste_cego": "2019-09-01 a 2026-06-30 (2.495 dias)",
  "metricas_medidas": {
    "kge": 0.11,
    "nse": -0.18,
    "rmse": 7.17,
    "mae": 4.69,
    "bias": 2.05,
    "f1_occ": 0.5469,
    "accuracy_occ": 0.7033,
    "brier_score": 0.204,
    "roc_auc": 0.764,
    "sdii_obs": 8.38,
    "sdii_pred": 8.01,
    "r10mm_obs": 287,
    "r10mm_pred": 610,
    "r20mm_obs": 100,
    "r20mm_pred": 13,
    "r20_recall_pct": 13.0,
    "qb95_bias_pct": -10.34,
    "qb99_bias_pct": -36.42,
    "cwd_obs": 29,
    "cwd_pred": 210
  }
}
```

### O Diagnóstico do V6d:
* **Ponto Forte**: SDII quase perfeito ($8.01$ vs $8.38\text{ mm}$) e QB95 aceitável ($-10.3\%$).
* **Pontos Fracos Críticos**:
  1. **R20 Recall = 13%**: De 100 dias de tempestade real $\ge 20\text{ mm}$, o modelo só previu 13.
  2. **CWD = 210 vs 29**: Garoa infinita (210 dias seguidos chovendo fraco quando o real foram 29).
  3. **R10 Overshoot**: 610 dias previstos contra 287 observados ($2.12\times$ over).

---

## 5. Matriz de Hipóteses de Projeto vs Resultados a Medir

No protocolo experimental do projeto (`docs/V7_HYBRID_ARCHITECTURE.md`), as metas do **V7 Híbrido** e do **V8 Espacial** são hipóteses científicas que o script `results_viewer.py` vai preencher com dados reais das execuções:

| Dimensão de Avaliação | V6d (Medido Real) | Meta de Hipótese V7 (Híbrida Temporal) | Meta de Hipótese V8 (Espacial Sinótica) |
| :--- | :---: | :---: | :---: |
| **KGE** | $0.11$ | Superar $>0.25$ via fusão multi-escala TCN | Superar $>0.35$ com informação de ventos/umidade |
| **R20 Recall (%)** | $13.0\%$ | Quebrar teto para $>35\%$ (Loss focada) | Atingir $>50\%$ (gatilho de velocidade vertical $w_{500}$) |
| **CWD (Garoa Consecutiva)** | $210$ dias | Reduzir para $<100$ dias | Reduzir para $<50$ dias |
| **QB99 Bias (%)** | $-36.4\%$ | Reduzir erro para $|\text{bias}| < 20\%$ | Reduzir erro para $|\text{bias}| < 15\%$ |
| **Física CC Real** | Falsa ($TCWV=20.0$ const.) | Falsa (permanece GT V2 para controle) | **Ativa com $TCWV$ real horário do ERA5** |

---

## 6. Como Auditar os Resultados com Seus Próprios Olhos

Cada treinamento executado no repositório grava uma linha bruta em `results/experiments_v7.jsonl`. 
Para compilar a tabela e abrir o painel interativo localmente:

```bash
python -m src.v7.results_viewer
```

Abra o arquivo gerado `results/DASHBOARD.html` no seu navegador. Ele lê diretamente o `.jsonl` gravado pela GPU, compara célula a célula contra a linha oficial do V6d acima e colore automaticamente:
* 🟢 **Verde**: Estatisticamente superior ao V6d.
* 🔴 **Vermelho**: Inferior ao V6d.
* 🟠 **Laranja**: Alerta de overshoot ou vício de variância.
