import csv, glob, sys, statistics as st

pattern = sys.argv[1] if len(sys.argv) > 1 else "perterm_GAAC_L5_*.csv"
for f in sorted(glob.glob(pattern)):
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        for k in ("delta","support_mean","ap_base","ap_fv"):
            r[k] = float(r[k])
        r["pos_seeds"]=int(r["pos_seeds"]); r["n_seeds"]=int(r["n_seeds"])
    d=[r["delta"] for r in rows]; n=len(rows)
    up=sum(1 for x in d if x>0); dn=sum(1 for x in d if x<0)
    up01=sum(1 for x in d if x>0.01); dn01=sum(1 for x in d if x<-0.01)
    ns = rows[0]["n_seeds"]
    cu=[r for r in rows if r["pos_seeds"]==r["n_seeds"]]
    cd=sum(1 for r in rows if r["pos_seeds"]==0)
    print(f"\n==== {f} ====")
    print(f"n_seeds={ns}  terms={n}  mean delta={st.mean(d):+.4f}  median={st.median(d):+.4f}")
    print(f"improved/degraded: {up}/{dn}  (coin-flip ~{n//2})")
    print(f">+0.01 / <-0.01: {up01}/{dn01}  (symmetry)")
    print(f"consistent all-up/all-down: {len(cu)}/{cd}  (chance ~{n*0.5**ns:.1f} each)")
    cu.sort(key=lambda r: r["delta"], reverse=True)
    for r in cu[:8]:
        gid=r["go_id"]; sup=r["support_mean"]; dl=r["delta"]; ab=r["ap_base"]; af=r["ap_fv"]
        print(f"   UP {gid:<12} sup={sup:.0f} d={dl:+.4f} ap {ab:.3f}->{af:.3f}")
