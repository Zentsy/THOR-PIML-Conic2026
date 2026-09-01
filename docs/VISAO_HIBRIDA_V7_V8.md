# THOR-PIML — A Nova Visão (era híbrida, 2026)

> **Documento-canônico da visão atual do projeto.** Para detalhes técnicos de arquitetura,
> loss e protocolo de validação: `docs/V7_HYBRID_ARCHITECTURE.md` (living doc).
> Para a história completa V1→V6d (diagnósticos, perdas e lições): `docs/archive/ESTADO_ATUAL_FINAL_V6.md`.
> Este documento substitui qualquer README/relatório anterior como fonte da verdade.

---

## 1. O que o projeto é hoje (uma frase)

**Downscaling de precipitação diária para Guarulhos-SP com modelo híbrido LSTM+TCN de duas
cabeças (hurdle: ocorrência × intensidade), treinado em CHIRPS 5.5 km + ERA5, com o objetivo
de corrigir vieses de projeções CORDEX 2026–2100.**

O nome THOR-PIML permanece; a sigla histórica ("Taylor-Hurdle...") caiu junto com a
arquitetura V1. O que sobreviveu de verdade de lá até aqui: o formato hurdle dual-head e a
ideia da física como restrição — tudo o mais foi reconstruído.

## 2. A mudança de pensamento: dois saltos, não um

O V6d bateu num teto diagnóstico (NSE −0.18, R20 recall 13%, CWD 210 vs 29) que a análise
final classificou como **informacional + arquitetural**. A resposta não foi "um modelo maior",
foi uma estratégia em dois saltos independentes e comparáveis:

| | **V7 — salto de ARQUITETURA** | **V8 — salto de INFORMAÇÃO** |
|---|---|---|
| Pergunta | Uma arquitetura híbrida quebra parte do teto? | A informação sinótica quebrará o resto? |
| O que muda | LSTM ‖ TCN multi-escala + fusão com portões + loss V7 (extremes-recall, variância) | + CNN 2D sobre domínio sinótico 6°×8° (z500, q700, w500, u/v700) + CAPE/TCWV reais + barreira Clausius-Clapeyron reativada |
| O que NÃO muda | Dados (GT V2), teste cego, protocolo | Arquitetura do tronco (herda a V7) |
| Papel no paper | **Controle**: isola o ganho de arquitetura | **Aposta principal**: atribui o ganho à informação |

**Por que isso importa:** se V7 ≈ V6d e V8 > ambos, o paper prova que o teto era informacional
— uma afirmação forte e útil para qualquer grupo tentando downscaling de ponto único. Se V7 já
ganhar, a decomposição mostra que era mistura. Derrota simples não existe nessa matriz: cada
linha da ablação responde uma pergunta.

## 3. Princípios não-negociáveis (herdados das falhas V1–V6)

1. **Extremos primeiro.** NSE sozinho é alvo de Goodhart. O projeto otimiza e julga por
   QB95/QB99, R10/R20 recall, SDII, CWD/CDD — o que importa para cheias e crises hídricas.
2. **Zero vazamento de dados.** Scaler fit só no treino; CV temporal em blocos expansivos;
   teste cego 2019-09→2026-06 intocado até a sentença final. Sem `year_norm`, sem LSTM bidirecional.
3. **Sem mock silencioso.** Qualquer script de dado falha alto se o dado real não existe
   (`THOR_ALLOW_MOCK=1` é a única porta). A V2 morreu por causa de um mock silencioso.
4. **Provenance de tudo.** Cada CSV carrega origem; `sources_metadata.json` é o registro;
   checagens de regressão bit-a-bit quando um dado é re-extraído (ex.: grade CHIRPS vs série
   original: max|Δ| = 0.0000 mm).
5. **Reprodutibilidade.** Seed fixa por padrão nos experimentos; checkpoints auto-contidos
   (config + scaler + features + git hash); toda run logada em JSONL estruturado.

## 4. Como o trabalho flui agora (o dia a dia)

```bash
# treino (na lightning.ai, T4/L4):
python -m src.v7.run --model hybrid|lstm|tcn --loss v7|v6d --data v2|v3 --cv 5 --seed 42 --num-workers 0

# dados da era V8 (Parte 2):
python data_prep/extract_chirps_grid.py          # ✅ feito — grade 30 células, check perfeito
python data_prep/fetch_era5_single.py --years 1981 2026   # CAPE/TCWV reais (em debug CDS)
python data_prep/fetch_era5_pl.py    --years 1981 2026    # domínio sinótico p/ CNN (em debug CDS)
python data_prep/build_gt_v3.py                 # merge → GT V3
python -m src.validate_datasets --gt-v3         # bateria física anti-mock

# V8 (Parte 3, quando GT V3 existir):
python -m src.v7.run_v8 --model hybrid --physics 0.05 --seed 42

# resultados — SEMPRE no final, cumulativo:
python -m src.v7.results_viewer   # → results/DASHBOARD.html (baseline V6d fixo, deltas coloridos)
```

**Registro de resultados:** cada run completa escreve (1) seu checkpoint nomeado pela config,
(2) uma linha em `results/experiments_v7.jsonl` (fonte da verdade), (3) a tabela
`results/experiments_v7.md` pronta pro paper. O dashboard lê o JSONL inteiro e compara tudo
contra o V6d. A CV é cacheada por fold (`results/cv_cache/`) — interrupção retoma de onde parou.

## 5. Estado real do projeto (2026-08-17)

**Pronto e verificado:**
- Pacote `src/v7/` completo: modelo híbrido (1.42M params), loss V7, pipeline zero-leakage,
  treino OneCycle+AMP+seleção-por-KGE, CLI com cache de CV; ablações lstm/tcn por flag.
- V8 codado e testado em sintético: CNN sinótica, pipeline de janelas pareadas, loss com
  barreira CC usando TCWV real.
- Dashboard de resultados + log estruturado + tutorial (`TUTORIAL_HIBRIDA_V7_V8.html`).
- Bugs V6 consertados na linha híbrida: lag-14 (V6 treinava 70 feats pensando que eram 84),
  `run_evaluation` com classe de modelo errada, ordem scheduler/optimizer no AMP,
  parsing de ano no extract_chirps_grid, requests CDS (date-range + chunks mensais).
- Grade CHIRPS 30 células extraída dos .nc locais com regressão bit-a-bit perfeita.

**Em andamento:**
- Parte 1 (ablações V7 sobre GT V2) rodando na lightning.ai — análise completa adiada para
  o fim, no dashboard (acumulativo).
- Downloads ERA5 (single + pressure levels) bloqueados em erro do lado CDS/MARS
  ("Duplicate value for month") — sonda de diagnóstico pronta; quando liberar, GT V3 sai em minutos.

**A seguir:**
1. Fechar downloads ERA5 → `build_gt_v3` → validação física.
2. V7 sobre GT V3 (a pergunta: "CAPE/TCWV reais já movem a agulha sem a CNN?").
3. V8 completo (CNN sinótica + física) — a aposta principal.
4. Matriz final no dashboard, CV no vencedor, inferência CORDEX (barato: <1h de GPU).

## 6. Onde está cada coisa

| Caminho | O quê |
|---|---|
| `src/v7/` | Era híbrida inteira (run, modelo, loss, pipeline, treino, dashboard) |
| `src/` (raiz) | Código V5/V6 legado — mantido para reavaliação do baseline V6d |
| `data_prep/` | Scripts de aquisição/construção (CHIRPS grade, ERA5 single/PL, GT V3) |
| `data/` | CSVs reais + provenance; `.nc` brutos nas pastas `*_netcdf/` |
| `checkpoints/`, `results/` | Artefatos por run; JSONL/dashboard/cv_cache |
| `docs/V7_HYBRID_ARCHITECTURE.md` | Referência técnica viva da era híbrida |
| `docs/PAPER_REFERENCES.md` | Bibliografia anotada para o artigo |
| `docs/archive/` | Toda a história V1→V6 (não usar como guia — usar como memória) |
