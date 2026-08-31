"""
THOR-PIML — Validador Completo de Datasets REAIS
Valida CHIRPS, ERA5-Land, CEMADEN, INMET e Ground Truth V2
Uso: python src/validate_datasets.py --all --verbose
"""
from pathlib import Path
import sys

# Garante UTF-8 no Windows PowerShell/CMD
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

try:
    from src.paths import DATA_DIR, CHECKPOINT_DIR
except ImportError:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

CHIRPS_CSV = DATA_DIR / "chirps_guarulhos_1981_2026.csv"
ERA5_CSV = DATA_DIR / "era5land_guarulhos_1981_2026.csv"
CEMADEN_CSV = DATA_DIR / "cemaden_guarulhos_2014_2026.csv"
GT_V2 = DATA_DIR / "ground_truth_guarulhos_daily_v2.csv"
GT_V1 = DATA_DIR / "ground_truth_guarulhos_daily.csv"
GT_V3 = DATA_DIR / "ground_truth_guarulhos_daily_v3.csv"
ERA5_PL_CSV = DATA_DIR / "era5pl_guarulhos_domain_1981_2026.csv"
ERA5_SINGLE_CSV = DATA_DIR / "era5_single_guarulhos_1981_2026.csv"
CHIRPS_GRID_STATS_CSV = DATA_DIR / "chirps_grid_stats_daily.csv"
SCALER_JSON = CHECKPOINT_DIR / "scaler_v2.json"

def check_file_exists(path, name, required=True):
    if not path.exists():
        return False, f"❌ {name} NÃO EXISTE: {path}", None
    size_kb = path.stat().st_size / 1024
    try:
        df_sample = pd.read_csv(path, nrows=3)
        cols = len(df_sample.columns)
    except:
        cols = 0
    try:
        n = len(pd.read_csv(path))
    except:
        n = 0
    return True, f"✅ {name} existe ({size_kb:.0f}KB, {n} linhas, cols={cols})", None

def validate_chirps(verbose=False):
    print("\n=== CHIRPS v2.0 5.5km (alvo) ===")
    ok, msg, _ = check_file_exists(CHIRPS_CSV, "CHIRPS")
    print(msg)
    if not ok:
        return False
    df = pd.read_csv(CHIRPS_CSV, parse_dates=["date"])
    issues=[]
    if "provenance" in df.columns:
        prov = df["provenance"].iloc[0]
        if "real" in str(prov).lower():
            print(f"  ✅ Proveniência REAL: {prov}")
        elif "mock" in str(prov).lower():
            print(f"  ❌ Proveniência MOCK: {prov}")
            issues.append("MOCK provenance")
    else:
        print(f"  ⚠️ Sem coluna provenance (antigo mock)")
    if "pr_chirps_mm" not in df.columns:
        print(f"  ❌ Falta coluna pr_chirps_mm"); issues.append("sem pr_chirps_mm")
    df = df.sort_values("date")
    date_diff = df["date"].diff().dt.days.dropna()
    gaps = (date_diff > 1).sum()
    if gaps>0:
        print(f"  ⚠️ {gaps} gaps de data >1 dia")
    else:
        print(f"  ✅ Continuidade OK")
    pr = df["pr_chirps_mm"]
    mean, p95, p99, pmax = pr.mean(), pr.quantile(0.95), pr.quantile(0.99), pr.max()
    r10, r20 = (pr>=10).sum(), (pr>=20).sum()
    zeros = (pr==0).mean()
    print(f"  📊 Stats: mean {mean:.2f}mm, p95 {p95:.1f}, p99 {p99:.1f}, max {pmax:.1f}, R10 {r10}, R20 {r20}, zeros {zeros:.1%}")
    if mean <1 or mean>10:
        print(f"  ⚠️ Mean {mean:.2f} fora esperado 1-10mm")
    else:
        print(f"  ✅ Mean plausível")
    if r20<100:
        print(f"  ⚠️ R20 {r20} baixo (real deve 800-1200)")
        issues.append("R20 baixo")
    else:
        print(f"  ✅ R20 plausível")
    if GT_V1.exists():
        try:
            v1 = pd.read_csv(GT_V1, usecols=["pr_target"])
            corr = np.corrcoef(v1["pr_target"].values[:len(pr)], pr.values[:len(v1)])[0,1]
            print(f"  📈 Corr NASA POWER V1: {corr:.3f} (real 0.6-0.85, mock 0.99)")
            if corr>0.98:
                print(f"  ❌ Corr alta, parece MOCK")
                issues.append("corr NASA >0.98")
        except Exception as e:
            if verbose:
                print(f"  Aviso corr: {e}")
    print(f"  {'✅ CHIRPS REAL OK' if len(issues)==0 else '❌ CHIRPS COM PROBLEMAS: '+', '.join(issues)}")
    return len(issues)==0

def validate_era5(verbose=False):
    print("\n=== ERA5-Land 9km + CAPE/TCWV ===")
    ok, msg, _ = check_file_exists(ERA5_CSV, "ERA5-Land")
    print(msg)
    if not ok:
        return False
    df = pd.read_csv(ERA5_CSV, parse_dates=["date"])
    issues=[]
    if "provenance" in df.columns:
        prov = df["provenance"].iloc[0]
        if "real" in str(prov).lower():
            print(f"  ✅ Proveniência REAL: {prov}")
        elif "mock" in str(prov).lower():
            print(f"  ❌ Proveniência MOCK: {prov}"); issues.append("MOCK")
    for col in ["tmean","tmax","tmin","rh","psfc","wind_speed","solar_rad"]:
        if col not in df.columns:
            print(f"  ❌ Falta {col}"); issues.append(f"sem {col}")
    for col in ["cape","tcwv"]:
        if col in df.columns:
            print(f"  ✅ V6 {col}: mean {df[col].mean():.1f}")
        else:
            print(f"  ⚠️ {col} não presente")
    checks=[("tmean",df["tmean"].min(),df["tmean"].max(),-10,45),("rh",df["rh"].min(),df["rh"].max(),0,100),("psfc",df["psfc"].min(),df["psfc"].max(),800,1050)]
    for name,mn,mx,emin,emax in checks:
        if name in df.columns:
            if mn<emin or mx>emax:
                print(f"  ⚠️ {name} fora range: {mn:.1f}-{mx:.1f}"); issues.append(f"{name} fora range")
    print(f"  📊 tmean {df['tmean'].mean():.1f}C, rh {df['rh'].mean():.0f}%, psfc {df['psfc'].mean():.0f}hPa")
    print(f"  {'✅ ERA5 REAL OK' if len(issues)==0 else '⚠️ ERA5 COM AVISOS: '+', '.join(issues)}")
    return True

def validate_cemaden(verbose=False):
    print("\n=== CEMADEN PCDs 2014+ ===")
    if not CEMADEN_CSV.exists():
        print(f"  ⚠️ CEMADEN não existe (ok, opcional)"); return True
    ok,msg,_=check_file_exists(CEMADEN_CSV,"CEMADEN"); print(msg)
    df=pd.read_csv(CEMADEN_CSV, parse_dates=["date"])
    pr_col="pr_cemaden_mm" if "pr_cemaden_mm" in df.columns else "pr_cemaden"
    mean,r20=df[pr_col].mean(),(df[pr_col]>=20).sum()
    print(f"  📊 mean {mean:.2f}mm R20 {r20}")
    if CHIRPS_CSV.exists():
        try:
            chirps=pd.read_csv(CHIRPS_CSV, parse_dates=["date"])
            merged=pd.merge(chirps,df,on="date",how="inner")
            if len(merged)>100 and "pr_chirps_mm" in merged.columns:
                corr=merged["pr_chirps_mm"].corr(merged[pr_col])
                print(f"  📈 Corr CHIRPS vs CEMADEN: {corr:.3f} (real 0.7-0.85, mock 0.99)")
                if corr>0.98: print(f"  ❌ Corr alta MOCK"); return False
        except Exception as e:
            if verbose: print(f"  Aviso corr: {e}")
    print(f"  ✅ CEMADEN OK")
    return True

def validate_ground_truth(verbose=False):
    print("\n=== Ground Truth V2 ===")
    ok,msg,_=check_file_exists(GT_V2,"Ground Truth V2"); print(msg)
    if not ok: return False
    df=pd.read_csv(GT_V2, parse_dates=["date"])
    issues=[]
    if "input_provenance" in df.columns:
        prov=df["input_provenance"].iloc[0]
        if "mock" in str(prov).lower():
            print(f"  ❌ input_provenance MOCK: {prov}"); issues.append("MOCK")
        elif "real" in str(prov).lower():
            print(f"  ✅ input_provenance REAL: {prov}")
    pr=df["pr_target"]; mean,r10,r20=pr.mean(),(pr>=10).sum(),(pr>=20).sum(); zeros=(pr==0).mean()
    print(f"  📊 mean {mean:.2f}mm R10 {r10} R20 {r20} zeros {zeros:.1%} {len(df)} dias")
    n=len(df); print(f"  📉 train {df['pr_target'].iloc[:int(n*0.7)].mean():.2f}mm val {df['pr_target'].iloc[int(n*0.7):int(n*0.85)].mean():.2f}mm test {df['pr_target'].iloc[int(n*0.85):].mean():.2f}mm")
    print(f"  {'✅ GT V2 REAL OK' if len(issues)==0 else '❌ GT V2 COM PROBLEMAS'}")
    return len(issues)==0

def validate_gt_v3(verbose=False):
    """V3: cape/tcwv REAIS + sinótica ERA5 PL + CHIRPS grid. Ranges físicos duros."""
    print("\n=== Ground Truth V3 (era híbrida) ===")
    if not GT_V3.exists():
        print(f"  ⚠️ GT V3 ainda não existe (rode data_prep/build_gt_v3.py na lightning.ai)")
        return True
    ok, msg, _ = check_file_exists(GT_V3, "Ground Truth V3")
    print(msg)
    df = pd.read_csv(GT_V3, parse_dates=["date"])
    issues = []
    prov = str(df["input_provenance"].iloc[0]) if "input_provenance" in df.columns else ""
    print(f"  📌 provenance: {prov}")
    if "mock" in prov.lower():
        print("  ❌ MOCK provenance"); issues.append("MOCK")

    range_checks = [
        # (col, min, max, é proxy se 1 único valor?)
        ("cape", 0.0, 6000.0, 300.0),   # J/kg real (proxy V2 ficava ~5-136)
        ("tcwv", 2.0, 80.0, 20.0),      # mm real (V2 constante 20.0)
        ("z500", 5000.0, 6200.0, None), # m geopotencial altura
        ("q700", 0.0, 20.0, None),      # g/kg
        ("w500", -5.0, 5.0, None),      # Pa/s
        ("ws700", 0.0, 40.0, None),     # m/s
        ("pr_grid_max", 0.0, 300.0, None),
        ("pr_grid_std", 0.0, 150.0, None),
    ]
    for col, lo, hi, proxy_val in range_checks:
        if col not in df.columns:
            print(f"  ⚠️ coluna {col} ausente (insumo faltou no build)")
            continue
        s = df[col].dropna()
        if len(s) == 0:
            print(f"  ⚠️ {col}: vazia"); continue
        nunique = s.nunique()
        flag = ""
        if proxy_val is not None and nunique <= 3:
            print(f"  ❌ {col}: {nunique} valores únicos — parece constante/proxy, não REAL")
            issues.append(f"{col} não-real")
            continue
        if s.min() < lo or s.max() > hi:
            print(f"  ❌ {col}: range {s.min():.2f}–{s.max():.2f} fora do físico [{lo},{hi}]")
            issues.append(f"{col} fora range")
            continue
        print(f"  ✅ {col}: {s.min():.2f}–{s.max():.2f} (nunique={nunique}){flag}")
    if "shear_700" in df.columns:
        s = df["shear_700"].dropna()
        if len(s):
            print(f"  ✅ shear_700: {s.min():.2f}–{s.max():.2f} m/s (ws700 − superfície)")
    # pr_target tem que ser IDÊNTICO ao V2 (comparabilidade)
    if GT_V2.exists():
        v2 = pd.read_csv(GT_V2, parse_dates=["date"])[["date", "pr_target"]]
        m = df[["date", "pr_target"]].merge(v2, on="date", suffixes=("_v3", "_v2"))
        if len(m) > 100:
            max_diff = (m["pr_target_v3"] - m["pr_target_v2"]).abs().max()
            status = "✅" if max_diff < 1e-6 else "❌"
            print(f"  {status} pr_target idêntico ao V2 (max|Δ|={max_diff:.2e}) — comparabilidade")
            if max_diff >= 1e-6:
                issues.append("pr_target mudou vs V2")
    print(f"  {'✅ GT V3 REAL OK' if len(issues) == 0 else '❌ GT V3 COM PROBLEMAS: ' + ', '.join(issues)}")
    return len(issues) == 0


def validate_scaler():
    print("\n=== Scaler V2 ===")
    if not SCALER_JSON.exists():
        print(f"  ⚠️ {SCALER_JSON} não existe"); return True
    import json
    data=json.loads(SCALER_JSON.read_text())
    print(f"  ✅ Scaler: method={data.get('method')} fitted={data.get('fitted')} features={len(data.get('feature_names',[]))}")
    return True

def main():
    import argparse
    parser=argparse.ArgumentParser(description="Validador datasets REAIS THOR-PIML")
    parser.add_argument("--all",action="store_true"); parser.add_argument("--chirps",action="store_true"); parser.add_argument("--era5",action="store_true"); parser.add_argument("--cemaden",action="store_true"); parser.add_argument("--ground-truth",action="store_true"); parser.add_argument("--gt-v3",action="store_true"); parser.add_argument("--scaler",action="store_true"); parser.add_argument("--verbose",action="store_true"); parser.add_argument("--fail-if-mock",action="store_true")
    args=parser.parse_args()
    if not any([args.all,args.chirps,args.era5,args.cemaden,args.ground_truth,args.gt_v3,args.scaler]): args.all=True
    print("="*70+"\nTHOR-PIML — Validador Datasets REAIS vs MOCK\n"+"="*70)
    results=[]
    if args.all or args.chirps: results.append(("CHIRPS",validate_chirps(args.verbose)))
    if args.all or args.era5: results.append(("ERA5",validate_era5(args.verbose)))
    if args.all or args.cemaden: results.append(("CEMADEN",validate_cemaden(args.verbose)))
    if args.all or args.ground_truth: results.append(("GT V2",validate_ground_truth(args.verbose)))
    if args.all or args.gt_v3: results.append(("GT V3",validate_gt_v3(args.verbose)))
    if args.all or args.scaler: results.append(("Scaler",validate_scaler()))
    print("\n"+"="*70+"\nRESUMO:")
    for name,ok in results:
        print(f"  {'✅' if ok else '❌'} {name}: {'REAL OK' if ok else 'FALHA/MOCK'}")
    all_ok=all(ok for _,ok in results)
    if all_ok:
        print("\n✅ TODOS REAIS OK — pode treinar V6")
    else:
        print("\n❌ ALGUM COM PROBLEMA — veja GUIA_DADOS_REAIS")
        if args.fail_if_mock:
            import sys; sys.exit(1)

if __name__=="__main__":
    main()
