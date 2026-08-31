# 🏆 Benchmark Oficial — THOR-PIML vs Modelos Canônicos da Literatura

> **Protocolo:** Teste Cego Independente (2019-09-02 a 2026-06-30) • **Ground Truth:** CHIRPS 5.5km (GT V3)


| Métrica Avaliada | Observado (Real) | EQM (Gudmundsson 2012) | THOR-V7 Híbrido (Proposta) | THOR-V8 Espacial (Proposta) | ResLSTM (Kratzert 2018) | TCN Pura (Bai 2018) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KGE (Kling-Gupta)** | 1.0000 | 0.0624 | 0.3200 | **0.4101** 🏆 | 0.1603 | 0.2383 |
| **NSE (Nash-Sutcliffe)** | 1.0000 | -0.9879 | -0.0939 | **-0.0336** 🏆 | -0.3013 | -0.3418 |
| **RMSE (mm/dia)** | 0.00 | 9.31 | 6.90 | **6.71** 🏆 | 7.53 | 7.64 |
| **MAE (mm/dia)** | 0.00 | 4.19 | 4.03 | **3.44** 🏆 | 4.62 | 4.41 |
| **Bias (mm/dia)** | +0.00 | +1.56 | +1.02 | **+0.14** 🏆 | +1.87 | +0.70 |
| **F1-Score Ocorrência** | 1.0000 | **0.7182** 🏆 | 0.4980 | 0.4103 | 0.4943 | 0.3682 |
| **Acurácia Ocorrência** | 100.00% | **80.39%** 🏆 | 70.44% | 69.64% | 69.76% | 66.79% |
| **SDII (mm/dia chuvoso)** | 8.39 | 16.22 | 8.05 | 6.37 | **8.41** 🏆 | 4.32 |
| **QB95% (Vício Quantil 95%)** | +0.0% | +61.2% | -3.6% | **+0.6%** 🏆 | +6.2% | +7.4% |
| **QB99% (Vício Quantil 99%)** | +0.0% | +78.2% | -34.0% | **-9.2%** 🏆 | -25.9% | -31.1% |
| **R10mm (Dias >= 10mm)** | 288 | 425 | 470 | **279** 🏆 | 544 | 380 |
| **R20mm (Dias >= 20mm)** | 100 | 206 | 17 | 95 | **97** 🏆 | 92 |
| **CWD (Máx. Dias Chuvosos)** | 29 | **25** 🏆 | 9 | 71 | 14 | 69 |

---

> **📌 Notas Metodológicas e Proveniência dos Códigos:**
> 1. **EQM (Gudmundsson et al. 2012 / Maraun 2013)\*:** Baseado no repositório `scikit-downscale` (Pangeo Data), adaptado com CDF empírica não-paramétrica de 2.000 quantis para compatibilidade nativa com scikit-learn moderno / Python 3.12.
> 2. **ResLSTM (Kratzert et al. 2018):** Arquitetura canônica de hidrologia neural do repositório oficial `neuralhydrology`.
> 3. **TCN Pura (Bai et al. 2018):** Arquitetura convolucional causal dilatada do repositório oficial CMU LocusLab `TCN`.
> 4. **THOR-V7 Híbrido (LSTM + TCN Gated Fusion):** Nossa proposta de fusão temporal informada por física.
> 5. **THOR-V8 Espacial (CNN 2D + Hybrid + PIML Master):** Nossa proposta master com acoplamento sinótico 2D e garantia física de Clausius-Clapeyron (0.00% de violação).