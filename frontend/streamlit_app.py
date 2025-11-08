# frontend/streamlit_app.py
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Deadstock Redistribution", page_icon="♻️", layout="wide")

st.markdown("""
<style>
html, body, [class^="css"] { font-size: 16px; line-height: 1.5; }
[data-testid="stSidebar"] * { font-size: 15px; }
.kpi { padding: 1rem; border: 1px solid #333; border-radius: 0.75rem; }
h1, h2, h3 { letter-spacing: .2px; }
</style>
""", unsafe_allow_html=True)

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")

with st.sidebar:
    st.title("Settings")
    api_url = st.text_input("Backend API URL", value=DEFAULT_API, key="set_api_url")
    timeout = st.number_input("Request timeout (s)", 5, 60, 15, key="set_timeout")

def api_get(path: str, params: dict | None = None):
    url = f"{api_url.rstrip('/')}/{path.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)

tab_overview, tab_find, tab_redis, tab_analytics, tab_whatif, tab_bundles, tab_help = st.tabs(
    ["Overview", "Search & Recommend", "Redistribution Plan", "Analytics",
     "What-If Simulation", "Bundles & CSR", "Help"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    st.markdown("# Deadstock Dilemma — AI Redistribution")

    msg, err = api_get("/")
    if err:
        st.error(f"Backend error: {err}")
    else:
        st.success(msg.get("message", "Backend reachable."))

    raw, err = api_get("/inventory", params={"limit": 500})
    if err or not isinstance(raw, list) or len(raw) == 0:
        st.warning("Inventory preview unavailable.")
        st.stop()

    df = pd.DataFrame(raw).rename(columns={
        "product_id":"Product ID","category":"Category","region":"Region",
        "inventory_level":"Inventory","units_sold":"Units sold","deadstock_flag":"Deadstock",
        "text_feature":"Text"
    })

    total = len(df)
    dead = int(df["Deadstock"].sum()) if "Deadstock" in df.columns else 0
    inv_proxy = int(df.get("Inventory", pd.Series([0]*len(df))).sum())
    dead_inv_proxy = int(df.loc[df.get("Deadstock", pd.Series(False)), "Inventory"].sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f'<div class="kpi"><h3>Items</h3><h2>{total:,}</h2></div>', unsafe_allow_html=True)
    with k2:
        pct = (dead/total*100) if total else 0
        st.markdown(f'<div class="kpi"><h3>Deadstock</h3><h2>{dead:,} ({pct:.1f}%)</h2></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="kpi"><h3>Inventory</h3><h2>{inv_proxy:,}</h2></div>', unsafe_allow_html=True)
    with k4: st.markdown(f'<div class="kpi"><h3>Deadstock Inventory</h3><h2>{dead_inv_proxy:,}</h2></div>', unsafe_allow_html=True)

    st.markdown("### Inventory")
    cats_list = sorted(df["Category"].dropna().unique()) if "Category" in df.columns else []
    regs_list = sorted(df["Region"].dropna().unique()) if "Region" in df.columns else []

    c1, c2, c3 = st.columns(3)
    f_cat = c1.multiselect("Category", options=cats_list, default=[], placeholder="All", key="ov_cat")
    f_reg = c2.multiselect("Region", options=regs_list, default=[], placeholder="All", key="ov_reg")
    f_dead = c3.checkbox("Deadstock only", value=False, key="ov_dead")

    fdf = df.copy()
    if f_cat: fdf = fdf[fdf["Category"].isin(f_cat)]
    if f_reg: fdf = fdf[fdf["Region"].isin(f_reg)]
    if f_dead and "Deadstock" in fdf.columns: fdf = fdf[fdf["Deadstock"] == True]

    order = ["Product ID","Category","Region","Inventory","Units sold","Deadstock"]
    order = [c for c in order if c in fdf.columns] + [c for c in fdf.columns if c not in order]
    st.dataframe(fdf[order], use_container_width=True, height=420)

    st.download_button(
        "Download CSV",
        data=fdf.to_csv(index=False),
        file_name="inventory_filtered.csv",
        mime="text/csv",
        key="ov_download"
    )

# ---------- SEARCH & RECOMMEND ----------
with tab_find:
    st.markdown("## Search")
    q = st.text_input("Keyword", placeholder="e.g., milk, bakery, toys", key="srch_kw")
    if st.button("Search", type="primary", key="srch_btn"):
        res, err = api_get("/search", params={"keyword": q.strip()})
        if err: st.error(err)
        else:
            items = res.get("results", [])
            if not items: st.info("No matches.")
            else:
                sdf = pd.DataFrame(items)
                order = ["product_id","category","region","inventory_level","units_sold","deadstock_flag","text_feature"]
                order = [c for c in order if c in sdf.columns] + [c for c in sdf.columns if c not in order]
                st.dataframe(sdf[order], use_container_width=True, height=380)

    st.markdown("## Recommendations")
    inv, _ = api_get("/inventory", params={"limit": 400})
    ids = [r.get("product_id") for r in inv] if isinstance(inv, list) else []
    pid = st.selectbox("Product ID", options=ids, index=0 if ids else None, key="recs_pid")
    topk = st.number_input("Top K", 1, 20, 5, step=1, key="recs_topk")

    if st.button("Get Recommendations", type="primary", key="recs_btn"):
        recs, err = api_get("/recommendations", params={"product_id": pid, "top_k": topk})
        if err: st.error(err)
        else:
            rows = recs.get("recommendations", [])
            if not rows: st.info("No recommendations.")
            else:
                rdf = pd.DataFrame(rows)
                if "Distance" in rdf.columns: rdf["similarity"] = (1 - rdf["Distance"]).round(3)
                order = ["Product_ID","Category","Region","Deadstock","similarity","Distance"]
                order = [c for c in order if c in rdf.columns] + [c for c in rdf.columns if c not in order]
                st.dataframe(rdf[order], use_container_width=True, height=360)
                st.download_button(
                    "Download CSV",
                    data=rdf.to_csv(index=False),
                    file_name=f"recs_{pid}.csv",
                    mime="text/csv",
                    key="recs_download"
                )

# ---------- REDISTRIBUTION PLAN ----------
with tab_redis:
    st.markdown("## Redistribution Plan")
    inv, _ = api_get("/inventory", params={"limit": 400})
    ids = [r.get("product_id") for r in inv] if isinstance(inv, list) else []
    pid = st.selectbox("Product ID", options=ids, index=0 if ids else None, key="rd_pid")

    c1, c2, c3 = st.columns(3)
    top_regions = c1.number_input("Target regions", 1, 10, 3, step=1, key="rd_targets")
    max_fraction = c2.slider("Max fraction to move", 0.1, 1.0, 0.5, step=0.1, key="rd_fraction")
    category_aware = c3.checkbox("Category-aware", value=True, key="rd_catware")
    tol = st.slider("Tolerance", 0.0, 0.30, 0.10, step=0.05, key="rd_tol")

    if st.button("Generate Plan", type="primary", key="rd_btn"):
        plan, err = api_get("/redistribution_plan", params={
            "product_id": pid, "top_regions": int(top_regions),
            "max_fraction": float(max_fraction),
            "category_aware": bool(category_aware),
            "tolerance_ratio": float(tol),
        })
        if err: st.error(err)
        else:
            meta = st.columns(5)
            meta[0].metric("Source", plan.get("source_region","—"))
            meta[1].metric("Category", plan.get("category","—"))
            meta[2].metric("Available qty", plan.get("available_qty","—"))
            meta[3].metric("Max transfer", plan.get("max_transfer_considered","—"))
            meta[4].metric("Tolerance used", plan.get("tolerance_used","—"))

            if "plan" not in plan or len(plan["plan"]) == 0:
                st.info(plan.get("reason_if_empty", "No suitable targets."))
                preview = plan.get("gap_preview", [])
                if preview:
                    st.dataframe(pd.DataFrame(preview), use_container_width=True, height=280)
                st.stop()

            rdf = pd.DataFrame(plan["plan"])
            rdf["similarity_%"] = (rdf["similarity"] * 100).round(1)
            st.dataframe(rdf[["target_region","suggested_qty","gap","similarity_%","reason"]],
                         use_container_width=True, height=360)
            st.download_button(
                "Download CSV",
                data=rdf.to_csv(index=False),
                file_name=f"redistribution_plan_{pid}.csv",
                mime="text/csv",
                key="rd_download"
            )

            try:
                flows = rdf.copy()
                flows["source"] = plan.get("source_region","Source")
                label = [flows["source"].iloc[0]] + flows["target_region"].tolist()
                src = [0]*len(flows)
                tgt = list(range(1, len(flows)+1))
                val = flows["suggested_qty"].tolist()
                fig = go.Figure(data=[go.Sankey(
                    node=dict(label=label, pad=20, thickness=18),
                    link=dict(source=src, target=tgt, value=val)
                )])
                fig.update_layout(title="Proposed stock flow", height=350)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# ---------- ANALYTICS ----------
with tab_analytics:
    st.markdown("## Analytics")
    raw, err = api_get("/inventory", params={"limit": 1000})
    if err or not isinstance(raw, list) or len(raw) == 0:
        st.warning("No data.")
    else:
        adf = pd.DataFrame(raw).rename(columns={
            "inventory_level":"Inventory","units_sold":"Units sold",
            "category":"Category","region":"Region","deadstock_flag":"Deadstock"
        })

        a1, a2 = st.columns(2)
        by_cat = adf.groupby("Category", dropna=False).agg(Inventory=("Inventory","sum"), Deadstock=("Deadstock","sum")).reset_index()
        a1.plotly_chart(px.bar(by_cat, x="Category", y="Inventory", title="Inventory by Category"), use_container_width=True)
        a2.plotly_chart(px.bar(by_cat, x="Category", y="Deadstock", title="Deadstock by Category"), use_container_width=True)

        b1, b2 = st.columns(2)
        by_reg = adf.groupby("Region", dropna=False).agg(Inventory=("Inventory","sum"), Sales=("Units sold","sum")).reset_index()
        b1.plotly_chart(px.bar(by_reg, x="Region", y=["Inventory","Sales"], title="Inventory vs Sales by Region"), use_container_width=True)
        adf["Sell-through"] = (adf["Units sold"]+1)/(adf["Inventory"]+1)
        b2.plotly_chart(px.box(adf, x="Category", y="Sell-through", title="Sell-through by Category"), use_container_width=True)

# ---------- WHAT-IF SIMULATION ----------
with tab_whatif:
    st.markdown("## What-If Simulation")

    base_raw, _ = api_get("/inventory", params={"limit": 2000})
    base_df = pd.DataFrame(base_raw) if isinstance(base_raw, list) else pd.DataFrame()
    ids = base_df["product_id"].astype(str).unique().tolist() if "product_id" in base_df.columns else []

    sim_pid = st.selectbox("Product", options=ids, index=0 if ids else None, key="sim_pid")
    c1, c2, c3 = st.columns(3)
    sim_targets = c1.number_input("Targets", 1, 10, 3, step=1, key="sim_targets")
    sim_frac = c2.slider("Max fraction", 0.1, 1.0, 0.4, step=0.1, key="sim_frac")
    sim_tol = c3.slider("Tolerance", 0.0, 0.30, 0.10, step=0.05, key="sim_tol")
    sim_cat = st.checkbox("Category-aware", value=True, key="sim_cat")

    if st.button("Run Simulation", type="primary", key="sim_btn"):
        if not sim_pid:
            st.warning("Choose a product.")
            st.stop()

        plan, err = api_get("/redistribution_plan", params={
            "product_id": sim_pid,
            "top_regions": int(sim_targets),
            "max_fraction": float(sim_frac),
            "category_aware": bool(sim_cat),
            "tolerance_ratio": float(sim_tol),
        })
        if err: st.error(err); st.stop()

        if base_df.empty:
            st.warning("No data.")
            st.stop()

        upd = base_df.rename(columns={
            "product_id":"Product ID","region":"Region","category":"Category",
            "inventory_level":"Inventory","units_sold":"Units sold","deadstock_flag":"Deadstock"
        }).copy()

        base_dead = int(upd.get("Deadstock", pd.Series(False)).sum()) if "Deadstock" in upd.columns else 0
        base_inv = int(upd.get("Inventory", pd.Series([0]*len(upd))).sum())

        if "plan" not in plan or len(plan["plan"]) == 0:
            st.dataframe(upd.head(20), use_container_width=True, height=300)
            st.stop()

        src = plan.get("source_region")
        moves = pd.DataFrame(plan["plan"])
        qty_moved = int(moves["suggested_qty"].sum())

        mask_src = (upd["Region"] == src) & (upd["Product ID"].astype(str) == str(sim_pid))
        if mask_src.any():
            upd.loc[mask_src, "Inventory"] = np.maximum(0, upd.loc[mask_src, "Inventory"] - qty_moved)

        for _, r in moves.iterrows():
            tgt_mask = (upd["Region"] == r["target_region"]) & (upd["Product ID"].astype(str) == str(sim_pid))
            if tgt_mask.any():
                upd.loc[tgt_mask, "Inventory"] = upd.loc[tgt_mask, "Inventory"] + int(r["suggested_qty"])
            else:
                new_row = {
                    "Product ID": sim_pid, "Category": plan.get("category","-"),
                    "Region": r["target_region"], "Inventory": int(r["suggested_qty"]),
                    "Units sold": 0, "Deadstock": False
                }
                upd = pd.concat([upd, pd.DataFrame([new_row])], ignore_index=True)

        if "Inventory" in upd.columns and "Units sold" in upd.columns:
            mean_inv = upd["Inventory"].mean() if len(upd) else 0
            mean_sales = upd["Units sold"].mean() if len(upd) else 0
            upd["Deadstock"] = (upd["Inventory"] > mean_inv * 1.5) & (upd["Units sold"] < mean_sales * 0.5)

        after_dead = int(upd["Deadstock"].sum()) if "Deadstock" in upd.columns else base_dead
        after_inv = int(upd["Inventory"].sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Inventory", after_inv, delta=after_inv - base_inv)
        m2.metric("Deadstock count", after_dead, delta=after_dead - base_dead)
        m3.metric("Qty moved", qty_moved)

        br = base_df.rename(columns={"inventory_level":"Inventory","region":"Region"}).groupby("Region").agg(Inventory=("Inventory","sum")).reset_index()
        ar = upd.groupby("Region").agg(Inventory=("Inventory","sum")).reset_index()
        br["State"] = "Before"; ar["State"] = "After"
        comb = pd.concat([br, ar])
        st.plotly_chart(px.bar(comb, x="Region", y="Inventory", color="State", barmode="group",
                               title="Regional inventory: Before vs After"), use_container_width=True)

        try:
            flows = moves.copy()
            flows["source"] = src
            label = [src] + flows["target_region"].tolist()
            src_idx = [0]*len(flows)
            tgt_idx = list(range(1, len(flows)+1))
            val = flows["suggested_qty"].tolist()
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=label, pad=20, thickness=18),
                link=dict(source=src_idx, target=tgt_idx, value=val)
            )])
            fig.update_layout(title="Simulated stock flow", height=350)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        st.dataframe(upd.sort_values(["Region","Product ID"]).head(30), use_container_width=True, height=360)

# ---------- BUNDLES & CSR ----------
with tab_bundles:
    st.markdown("## Bundles & CSR")

    inv, _ = api_get("/inventory", params={"limit": 400})
    ids = [r.get("product_id") for r in inv] if isinstance(inv, list) else []

    st.subheader("Bundle Builder")
    b_pid = st.selectbox("Anchor product", options=ids, index=0 if ids else None, key="bd_pid")
    k = st.slider("Partners", 1, 10, 4, key="bd_k")
    if st.button("Suggest Bundle", key="bd_btn"):
        recs, err = api_get("/recommendations", params={"product_id": b_pid, "top_k": k})
        if err: st.error(err)
        else:
            dfb = pd.DataFrame(recs.get("recommendations", []))
            if dfb.empty: st.info("No partners.")
            else:
                if "Distance" in dfb.columns: dfb["similarity"] = (1 - dfb["Distance"]).round(3)
                st.dataframe(dfb[["Product_ID","Category","Region","Deadstock","similarity"]],
                             use_container_width=True, height=300)

    st.markdown("---")

    st.subheader("Donation / CSR Routing")
    d_pid = st.selectbox("Product to donate", options=ids, index=0 if ids else None, key="don_pid")
    d_qty = st.number_input("Quantity", 1, 10000, 50, key="don_qty")
    regions = ["North","South","East","West","Central"]
    preferred_region = st.selectbox("Region", options=regions, index=0, key="don_region")

    plist, err = api_get("/partners", params={"region": preferred_region})
    if err:
        st.error(err)
    else:
        partners = pd.DataFrame(plist.get("partners", []))
        if partners.empty:
            st.warning("No partners configured for this region.")
        else:
            cols = [c for c in ["ngo","contact","email","phone","address","min_qty","notes"] if c in partners.columns]
            st.dataframe(partners[cols], use_container_width=True, height=220)
            partner_names = partners["ngo"].tolist() if "ngo" in partners.columns else []
            partner_choice = st.selectbox("Partner", options=partner_names, index=0 if partner_names else None, key="don_partner")
            notes = st.text_area("Notes", key="don_notes")

            if st.button("Record Donation", type="primary", key="don_btn"):
                try:
                    row = partners[partners["ngo"] == partner_choice].iloc[0].to_dict()
                    payload = {
                        "product_id": d_pid,
                        "qty": int(d_qty),
                        "region": preferred_region,
                        "partner_ngo": partner_choice,
                        "partner_contact": row.get("contact"),
                        "partner_email": row.get("email"),
                        "notes": notes or None
                    }
                    url = f"{api_url.rstrip('/')}/donate"
                    r = requests.post(url, json=payload, timeout=15)
                    r.raise_for_status()
                    resp = r.json()
                    conf = pd.DataFrame([resp["recorded"]])
                    st.success("Donation recorded.")
                    st.download_button(
                        "Download confirmation",
                        data=conf.to_csv(index=False).encode("utf-8"),
                        file_name=f"donation_{d_pid}_{preferred_region}.csv",
                        mime="text/csv",
                        key="don_download"
                    )
                except requests.RequestException as e:
                    st.error(f"Failed: {e}")

# ---------- HELP ----------
with tab_help:
    st.markdown("## Help")
    st.write("""
Use the tabs from left to right:
1) **Overview** – quick KPIs and inventory view  
2) **Search & Recommend** – find items and similar products  
3) **Redistribution Plan** – suggested transfers by region  
4) **Analytics** – category/region charts  
5) **What-If Simulation** – visualize impact before acting  
6) **Bundles & CSR** – bundle ideas and donation routing
""")
