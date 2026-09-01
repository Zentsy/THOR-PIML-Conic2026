# THOR-PIML — Bibliografia Anotada para o Paper Final

> Gerado em 2026-08-15 (busca web + conhecimento de domínio). Status:
> - ✅ = verificado via busca (título/venue conferidos no resultado)
> - ⚠️ = da memória do modelo — **conferir DOI/autores antes de citar**
> Nada aqui é citação final; é o mapa de leitura + lista de candidatos.

## 1. Reviews de DL para downscaling (estrutura do related work)

| Ref | Ideia-chave | Relevância | Status |
|---|---|---|---|
| [Are DL Methods Suitable for Downscaling GCMs?](https://journals.ametsoc.org/view/journals/aies/4/4/AIES-D-24-0121.1.xml) (AMS AIES) | Review + avaliação de DeepESD (CNN) | molda a seção "state of the art" | ✅ |
| [ML in Climate Downscaling: Critical Review](https://www.mdpi.com/2073-4441/18/2/271) (Water, 2025) | Síntese 2010-2025, estatístico→DL | panorama recente, gap regional | ✅ |
| [DL in statistical downscaling for gridded meteorological data](https://www.sciencedirect.com/science/article/abs/pii/S0924271623003489) (ISPRS J.) | Review sistemático de DL p/ grade | metodologia comparável | ✅ |
| [Exploring ML approaches for precipitation downscaling](https://www.tandfonline.com/doi/full/10.1080/10095020.2025.2477547) (Geo-spat. Inf. Sci., 2025; [PDF](https://opus.bibliothek.uni-augsburg.de/opus4/files/127090/127090.pdf)) | Review crítica de downscaling espacial | limites de super-resolução | ✅ |
| [Pan-European DL downscaling](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000630) (JGR ML, 2025) | CNN profunda EU, temp+precip | comparável europeu | ✅ |
| [Limitation of super-resolution ML downscaling](https://www.nature.com/articles/s41598-025-05880-7) (Sci Rep, 2025) | 100→12.5km, onde SR falha | justifica hurdle + física | ✅ |

## 2. Arquiteturas CNN-LSTM / espaciais (base da híbrida)

| Ref | Ideia-chave | Relevância | Status |
|---|---|---|---|
| Vandal et al., "DeepSD" (KDD 2017) | SR de GCM p/ precip | clássico downscaling CNN | ⚠️ |
| Shi et al., ConvLSTM nowcasting (NeurIPS 2017) | precip nowcasting, conv + recorrência | ancestral da híbrida | ⚠️ |
| Ravuri et al., DGMR (Nature 2021) | generativo radar nowcasting | qualidade de extremos | ⚠️ |
| Zhang et al., NowcastNet (Nature 2023) | física + generativo extremos | PIML de precip | ⚠️ |
| MetNet / MetNet-2 (Google, 2020/21) | CNN-LSTM global nowcasting | escala arquitetural | ⚠️ |
| [BMA + U-Net downscaling](https://www.sciencedirect.com/science/article/abs/pii/S1364815225003664) (2025) | U-Net + BMA, corr 0.68→0.82 | CNN espacial regional | ✅ |
| [LSTM downscaling framework, Austrália](https://neurips.cc/virtual/2023/76878) (NeurIPS WS 2023) | LSTM diário com ERA5 | setup mais próximo do nosso | ✅ |
| [RainBench / RTransUNet](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003120) (JAMES 2022) | generativo 10× resolução ERA5 | era5-based SR | ✅ |
| [Hybrid CNN-LSTM rainfall](https://www.mdpi.com/1999-4893/19/5/394) (Algorithms 2026) | CNN front-end + LSTM | híbrida direta ( recente) | ✅ |

## 3. Zero-inflated / hurdle / extremos (a nossa perna estatística)

| Ref | Ideia-chave | Relevância | Status |
|---|---|---|---|
| [DEMM (KDD 2022)](https://www.cse.msu.edu/~ptan/papers/kdd2022.pdf) | hurdle DL + mistura p/ cauda pesada | MAIS próximo do THOR | ✅ |
| [Hurdle–IMDL (arXiv 2025)](https://arxiv.org/html/2510.20486v1) | hurdle + aprendizado desbalanceado precip | estado da arte hurdle | ✅ |
| [Diffusion zero-inflated precip](https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA250921) (2025) | hurdle não diferencia origem dos zeros | discussão limitações | ✅ |
| Beckmann & ?., Bernoulli-gamma GLM downscaling | hurdle clássico climatológico | base estatística | ⚠️ |

## 4. Física-informed (a nossa perna PIML)

| Ref | Ideia-chave | Relevância | Status |
|---|---|---|---|
| Raissi et al., PINNs (JCP 2019) | fundação PIML | citação obrigatória | ⚠️ |
| Allen & Ingram, Nature 2002 | CC scaling ~7%/K extremos | base da barreira | ⚠️ |
| O'Gorman / Pfahl 2017 | scaling de extremos precip | W_max defensável | ⚠️ |
| Clausius-Clapeyron em Tetens/Magnus | formulação e_s | apêndice métodos | ⚠️ |

## 5. Sequência temporal: TCN vs LSTM (justifica o ramo TCN)

| Ref | Ideia-chave | Relevância | Status |
|---|---|---|---|
| Bai et al. 2018, "Empirical Evaluation TCN vs RNN" | TCN ≥ LSTM em série temporal | JUSTIFICATIVA CENTRAL do ramo TCN | ⚠️ ([leitura](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/)) |
| [Dilated causal convs overview](https://www.emergentmind.com/topics/dilated-causal-convolution-networks) | campo receptivo exponencial | desenho dos dilations | ✅ |

## 6. Dados & métricas (methods)

| Ref | Uso | Status |
|---|---|---|
| Hersbach et al. 2020, ERA5 (QJRMS) | reanálise preditores | ⚠️ |
| Funk et al. 2015, CHIRPS (Sci Data) | alvo 5.5km | ⚠️ |
| Giorgi et al. 2009, CORDEX | projeções futuras | ⚠️ |
| Zhang et al. 2011, ETCCDI indices (WIREs Clim Chg) | R10/R20/SDII/CWD/CDD | ⚠️ |
| Nash & Sutcliffe 1970; Gupta et al. 2009 (KGE) | NSE/KGE | ⚠️ |
| WMO 1mm rain-day threshold | hurdle 1.0mm | ⚠️ |

## 7. Regional (SE Brasil / São Paulo — gap que o paper preenche)

| Ref | Ideia-chave | Status |
|---|---|---|
| [Ramírez et al. 2006](https://journals.ametsoc.org/view/journals/wefo/21/6/waf981_1.xml) | downscaling NN estações SP/RJ/MG (77 cites) | ✅ |
| [Valverde et al. 2014](https://www.sciencedirect.com/science/article/abs/pii/S1568494614000957) | NN+fuzzy diário SP (35 cites) | ✅ |
| [da Silva et al. 2025, Chaos](https://web.if.usp.br/controle/sites/portal.if.usp.br.ifusp/files/Silva%2520Chaos%25202025.pdf) | ML chuva nas 5 regiões BR | ✅ |
| → **Gap:** ninguém combina ERA5(PL+single) + CHIRPS 5.5km + hurdle PIML em Guarulhos/SE-Brasil | nossa contribuição | — |

## 8. Filosofia da loss (introdução/discussão do paper)

- Kahneman & Tversky — Prospect Theory (loss aversion) → por que MSE simétrico vira "medíocre"
- Savage — Flaw of Averages
- Goodhart / Campbell — otimizar NSE-MSE destrói extremos
- (já referenciados no docs/ESTADO_ATUAL_FINAL_V6.md §loss)

## Estrutura sugerida do paper (IMO)

1. Intro: extremos em megacidades SE-Brasil; gap regional §7
2. Data: CHIRPS 5.5km alvo + ERA5(-Land/PL/single) §6
3. Method: hurdle dual-head + híbrida LSTM||TCN (§2, §5) + física CC (§4) + loss assimétrica (§3, §8)
4. Experiments: ablação (lstm/tcn/híbrida × surface/sinótica/espacial × loss v6d/v7) com CV temporal bloqueada + teste cego 2019-2026
5. Results: ETCCDI foco (QB95/QB99, R20 recall, SDII, CWD) — não NSE isolado (Goodhart)
6. CORDEX 2026-2100 (infer_cordex.py) como aplicação
