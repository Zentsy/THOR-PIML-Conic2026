#!/usr/bin/env python3
"""
========================================================================================
THOR-PIML — Script Canônico de Reprodução dos Resultados do Artigo (CONIC 2026)
========================================================================================
Este script reproduz em um único comando todas as tabelas e figuras oficiais
publicadas no artigo científico:

  1. Avalia os 5 modelos no Teste Cego Independente de 7 Anos (2.494 dias, 2019-2026):
     - Empirical Quantile Mapping (EQM) — Gudmundsson et al. (2012)
     - ResLSTM — Kratzert et al. (2018)
     - TCN — Bai et al. (2018)
     - THOR-V7 (Híbrido Temporal)
     - THOR-V8 (PIML Espaço-Temporal) — Campeão Oficial
  2. Gera as 5 Figuras Canônicas em Alta Resolução (300 DPI PNG + PDF Vetorial):
     - Fig 1: Tabela Editorial do Benchmark Oficial (Scoreboard Multicritério)
     - Fig 2: Ciclo Sazonal e Balanço de Volume por Estação (DJF / JJA)
     - Fig 3: Curvas de Permanência (FDC) e Preservação de Extremos (Q90–Q99.9)
     - Fig 4: Diagrama de Taylor Multicritério (Correlação, Variabilidade e RMSE)
     - Fig 5: Densidade Hexbin de Dispersão e Calibração Convectiva 1:1

Uso:
  python reproduce_paper_results.py             # Executa benchmark e gera todas as figuras
  python reproduce_paper_results.py --figures   # Gera apenas as figuras a partir dos resultados
  python reproduce_paper_results.py --benchmark # Executa apenas a bateria de testes numéricos
========================================================================================
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Configuração de encoding UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
CKPT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def print_banner():
    banner = """
========================================================================================
⚡ THOR-PIML: REPRODUÇÃO CANÔNICA DE RESULTADOS CIENTÍFICOS (CONIC 2026)
   Physics-Informed ML para Downscaling Regional de Extremos Climáticos (Guarulhos-SP)
========================================================================================
"""
    print(banner)


def check_prerequisites():
    """Valida a existência dos dados canônicos e checkpoints pré-treinados."""
    print("🔍 [1/3] Verificando pré-requisitos e integridade dos checkpoints...")
    
    required_files = [
        (DATA_DIR / "ground_truth_guarulhos_daily_v3.csv", "Dataset Ground Truth V3 (CHIRPS + ERA5)"),
        (CKPT_DIR / "v8_hybrid_seed42.pt", "Checkpoint Campeão: THOR-V8 Espacial PIML"),
        (CKPT_DIR / "v7_hybrid_v7_v3_seed42.pt", "Checkpoint: THOR-V7 Híbrido Temporal"),
        (CKPT_DIR / "v7_lstm_v7_v3_seed42.pt", "Checkpoint: ResLSTM (Kratzert 2018)"),
        (CKPT_DIR / "v7_tcn_v7_v3_seed42.pt", "Checkpoint: TCN (Bai 2018)"),
    ]
    
    missing = []
    for fpath, desc in required_files:
        if not fpath.exists():
            missing.append(f"  ❌ Faltando: {desc} ({fpath.name})")
        else:
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {desc} [{size_mb:.2f} MB]")

    if missing:
        print("\n[ERRO DE INTEGRIDADE]")
        print("\n".join(missing))
        print("\nPor favor, garanta que os checkpoints estejam baixados no diretório checkpoints/.")
        sys.exit(1)
        
    print("✓ Todos os dados e pesos pré-treinados estão presentes e verificados!\n")


def run_benchmark():
    """Executa a bateria de avaliação comparativa dos 5 modelos."""
    print("📊 [2/3] Executando benchmark oficial no teste cego de 7 anos (2.494 dias)...")
    from benchmarks.run_benchmarks import main as benchmark_main
    benchmark_main()
    print("✓ Benchmark numérico concluído com sucesso!\n")


def run_figures():
    """Gera as 5 figuras editoriais em 300 DPI."""
    print("🎨 [3/3] Renderizando as 5 figuras científicas oficiais em 300 DPI...")
    from src.generate_paper_figures import main as figures_main
    figures_main()
    print("\n✓ Todas as figuras foram geradas em 'results/figures/' em formatos PNG e PDF!")


def print_final_summary():
    """Exibe o resumo das saídas prontas para o paper."""
    print("\n" + "=" * 88)
    print("✅ REPRODUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 88)
    print("\n📁 Arquivos gerados para submissão no artigo:")
    print(f"  • Tabela Oficial do Benchmark:  {RESULTS_DIR / 'BENCHMARK_FINAL.md'}")
    print(f"  • Dashboard Interativo:          {RESULTS_DIR / 'BENCHMARK_DASHBOARD.html'}")
    print(f"  • Masterclass das Figuras:       {ROOT_DIR / 'docs' / 'AULA_FIGURAS_BENCHMARK.html'}")
    print(f"  • Galeria de Figuras (300 DPI):  {FIGURES_DIR}")
    print("    ├── fig1_benchmark_table_docx.png (.pdf)")
    print("    ├── fig2_seasonal_climatology_narrative.png (.pdf)")
    print("    ├── fig3_extremes_duration_curves.png (.pdf)")
    print("    ├── fig4_taylor_diagram.png (.pdf)")
    print("    └── fig5_convective_density_scatter.png (.pdf)")
    print("\n💡 Para visualizar o guia didático das figuras, abra no navegador:")
    print(f"   file://{ROOT_DIR / 'docs' / 'AULA_FIGURAS_BENCHMARK.html'}")
    print("=" * 88 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Script Canônico de Reprodução dos Resultados do Artigo THOR-PIML (CONIC 2026)"
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Gera exclusivamente as 5 figuras oficiais para o paper.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Executa exclusivamente o benchmark numérico dos modelos.",
    )
    args = parser.parse_args()

    t0 = time.time()
    print_banner()
    check_prerequisites()

    if args.figures:
        run_figures()
    elif args.benchmark:
        run_benchmark()
    else:
        run_benchmark()
        run_figures()

    elapsed = time.time() - t0
    print_final_summary()
    print(f"⏱️ Tempo total de execução: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
