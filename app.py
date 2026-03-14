import streamlit as st
import json
import urllib.parse
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="CloudFront Bot Log Analyzer", layout="wide")
st.title("CloudFront Bot Log Analyzer")

# -----------------------
# Helpers
# -----------------------
def safe_unquote(s: str) -> str:
    if s is None:
        return ""
    try:
        return urllib.parse.unquote_plus(str(s))
    except Exception:
        return str(s)

def classify_bot(ua: str) -> str:
    u = (ua or "").lower()
    # AI/LLM bots
    if "oai-searchbot" in u: return "OAI-SearchBot"
    if "gptbot" in u: return "GPTBot"
    if "chatgpt-user" in u: return "ChatGPT-User"

    # Search engines / major crawlers
    if "googlebot" in u: return "Googlebot"
    if "adsbot-google" in u: return "AdsBot-Google"
    if "bingbot" in u: return "Bingbot"
    if "applebot" in u: return "Applebot"
    if "yandex" in u: return "Yandex"
    if "duckduckbot" in u: return "DuckDuckBot"
    if "baiduspider" in u: return "Baiduspider"

    # SEO tools
    if "ahrefsbot" in u: return "AhrefsBot"
    if "semrushbot" in u: return "SemrushBot"
    if "mj12bot" in u or "majestic" in u: return "Majestic"
    if "dotbot" in u: return "DotBot"

    # Social previews
    if "facebookexternalhit" in u or "facebot" in u: return "Meta"
    if "twitterbot" in u: return "Twitterbot"

    # Generic bot patterns
    if any(k in u for k in [" bot", "bot/", "crawler", "spider", "slurp", "crawl"]):
        return "OtherBot"

    return "Non-bot/Browser/App"

def path_group(path: str) -> str:
    if not path: return "other"
    if path.startswith("/article/"): return "/article/*"
    if path.startswith("/tags/"): return "/tags/*"
    if path.startswith("/section/"): return "/section/*"
    if path.startswith("/live"): return "/live*"
    if path.startswith("/api/"): return "/api/*"
    if path.startswith("/static/"): return "/static/*"
    if path.startswith("/_next/"): return "/_next/*"
    if path.startswith("/images/"): return "/images/*"
    if path == "/robots.txt": return "/robots.txt"
    if path == "/": return "/ (homepage)"
    return "other"

def counter_to_df(counter: Counter, value_col="count"):
    df = pd.DataFrame(counter.items(), columns=["key", value_col])
    df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    total = df[value_col].sum() if len(df) else 1
    df["share_%"] = (df[value_col] / total * 100).round(2)
    return df

def plot_bar(df, x, y, title, rotate=45):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df[x].astype(str), df[y].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

def plot_stacked_share(df, index_col, category_col, value_col, title, top_n=12):
    """
    Stacked bar (share %) for top N index values by total requests.
    """
    if df.empty:
        st.info(f"No data for: {title}")
        return

    totals = df.groupby(index_col)[value_col].sum().sort_values(ascending=False).head(top_n)
    d = df[df[index_col].isin(totals.index)]

    pivot = d.pivot_table(index=index_col, columns=category_col, values=value_col, aggfunc="sum", fill_value=0)
    pivot = pivot.loc[totals.index]
    pivot_share = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = None
    for col in pivot_share.columns:
        vals = pivot_share[col].values
        if bottom is None:
            ax.bar(pivot_share.index.astype(str), vals, label=str(col))
            bottom = vals
        else:
            ax.bar(pivot_share.index.astype(str), vals, bottom=bottom, label=str(col))
            bottom = bottom + vals

    ax.set_title(title)
    ax.set_xlabel(index_col)
    ax.set_ylabel("Share within " + index_col + " (%)")
    plt.xticks(rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

def bot_insights(df_bots, df_status, df_xhosts, df_status_by_bot, df_pathgroup_by_bot, df_errors):
    total = int(df_status["requests"].sum()) if not df_status.empty else 0

    def get_val(df, col, key, val_col):
        r = df[df[col] == key]
        return int(r.iloc[0][val_col]) if not r.empty else 0

    http200 = get_val(df_status, "status", 200, "requests")
    http301 = get_val(df_status, "status", 301, "requests")
    http304 = get_val(df_status, "status", 304, "requests")
    err_total = int(df_errors["requests"].sum()) if (df_errors is not None and not df_errors.empty) else 0

    www = get_val(df_xhosts, "x-host-header", "www.almashhad.com", "requests")
    nonwww = get_val(df_xhosts, "x-host-header", "almashhad.com", "requests")

    def pct(n, d): return round(n/d*100, 2) if d else 0.0

    insights = []
    actions = []

    # Overall redirects
    redir_rate = pct(http301, total)
    if redir_rate >= 5:
        insights.append(f"High 301 redirect rate: {redir_rate}% (301={http301}).")
        actions.append("P0: Reduce redirects by enforcing a single canonical host (https + www) and removing redirect chains.")
    elif redir_rate >= 2:
        insights.append(f"Moderate 301 redirect rate: {redir_rate}%.")
        actions.append("P1: Audit top 301 URLs and remove avoidable redirects (host, slash, encoding).")

    # Host split
    nonwww_rate = pct(nonwww, www + nonwww)
    if nonwww_rate >= 2:
        insights.append(f"Non-www host usage: {nonwww_rate}% (`almashhad.com`) – likely generating redirects.")
        actions.append("P0: Normalize everything to https://www.almashhad.com (sitemaps, canonicals, internal links, OG).")

    # Googlebot
    if not df_status_by_bot.empty:
        gb_total = int(df_status_by_bot[df_status_by_bot["bot"] == "Googlebot"]["requests"].sum())
        if gb_total:
            gb_200 = int(df_status_by_bot[(df_status_by_bot["bot"] == "Googlebot") & (df_status_by_bot["status"] == 200)]["requests"].sum())
            gb_301 = int(df_status_by_bot[(df_status_by_bot["bot"] == "Googlebot") & (df_status_by_bot["status"] == 301)]["requests"].sum())
            gb_304 = int(df_status_by_bot[(df_status_by_bot["bot"] == "Googlebot") & (df_status_by_bot["status"] == 304)]["requests"].sum())
            insights.append(f"Googlebot status mix: 200={pct(gb_200, gb_total)}%, 301={pct(gb_301, gb_total)}%, 304={pct(gb_304, gb_total)}% (n={gb_total}).")
            if pct(gb_301, gb_total) >= 10:
                actions.append("P0: Ensure Googlebot lands on canonical URLs directly (www+https).")
            if pct(gb_304, gb_total) >= 30 and pct(gb_301, gb_total) >= 10:
                actions.append("P1: Review caching headers/ETag/Last-Modified; high revalidation + redirects suggests crawl inefficiency.")

    # Errors
    if err_total:
        insights.append(f"4xx/5xx is low: {pct(err_total, total)}% (count={err_total}).")
        top_err = df_errors.iloc[0]
        actions.append(f"P2: Fix top recurring error path: {top_err['path']} (hits={int(top_err['requests'])}).")

    # De-dupe actions
    seen = set()
    actions = [a for a in actions if not (a in seen or seen.add(a))]

    return insights, actions

# -----------------------
# UI
# -----------------------
uploaded_files = st.file_uploader(
    "Upload up to 5 CloudFront JSONL files (uncompressed)",
    type=["jsonl", "json", "txt"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 5:
    st.error("Please upload a maximum of 5 files at a time.")
    st.stop()

run = st.button("Run analysis")

if run and uploaded_files:
    # Aggregators
    total_requests = 0
    bot_counts = Counter()
    status_counts = Counter()
    method_counts = Counter()
    host_counts = Counter()
    xhost_counts = Counter()

    status_by_bot = defaultdict(Counter)
    pathgroup_by_bot = defaultdict(Counter)
    url_counts_by_bot = defaultdict(Counter)

    error_paths = Counter()
    error_samples = []

    progress = st.progress(0)
    total_files = len(uploaded_files)

    for i, uf in enumerate(uploaded_files, start=1):
        for raw in uf:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            total_requests += 1

            ua = safe_unquote(rec.get("cs(User-Agent)", ""))
            bot = classify_bot(ua)

            path = rec.get("cs-uri-stem", "") or ""
            host = rec.get("cs(Host)", "") or ""
            xhost = rec.get("x-host-header", "") or ""
            method = rec.get("cs-method", "") or ""

            # Status
            try:
                status = int(rec.get("sc-status", None))
            except Exception:
                status = None

            bot_counts[bot] += 1
            host_counts[host] += 1
            xhost_counts[xhost] += 1
            method_counts[method] += 1

            if status is not None:
                status_counts[status] += 1
                status_by_bot[bot][status] += 1

                if 400 <= status <= 599:
                    error_paths[path] += 1
                    if len(error_samples) < 30:
                        error_samples.append({
                            "status": status,
                            "path": path,
                            "bot": bot,
                            "ua": ua[:140],
                        })

            pg = path_group(path)
            pathgroup_by_bot[bot][pg] += 1

            if path:
                url_counts_by_bot[bot][path] += 1

        progress.progress(i / total_files)

    st.success(f"Processed {total_requests:,} requests.")

    # -----------------------
    # Build DataFrames (ALL)
    # -----------------------
    df_bots = counter_to_df(bot_counts, "requests").rename(columns={"key": "bot"})
    df_status = counter_to_df(status_counts, "requests").rename(columns={"key": "status"}).sort_values("status")
    df_methods = counter_to_df(method_counts, "requests").rename(columns={"key": "method"})
    df_hosts = counter_to_df(host_counts, "requests").rename(columns={"key": "cs(Host)"})
    df_xhosts = counter_to_df(xhost_counts, "requests").rename(columns={"key": "x-host-header"})

    # Status by bot
    rows = []
    for b, c in status_by_bot.items():
        total_b = sum(c.values()) or 1
        for stt, cnt in c.items():
            rows.append({"bot": b, "status": stt, "requests": cnt, "share_within_bot_%": round(cnt/total_b*100, 2)})
    df_status_by_bot = pd.DataFrame(rows)

    # Path groups by bot
    rows = []
    for b, c in pathgroup_by_bot.items():
        total_b = sum(c.values()) or 1
        for pg, cnt in c.items():
            rows.append({"bot": b, "path_group": pg, "requests": cnt, "share_within_bot_%": round(cnt/total_b*100, 2)})
    df_pathgroup_by_bot = pd.DataFrame(rows)

    # Top URLs by bot (Top N per bot)
    TOP_N = 50
    rows = []
    for b, c in url_counts_by_bot.items():
        for p, cnt in c.most_common(TOP_N):
            rows.append({"bot": b, "path": p, "requests": cnt})
    df_top_urls_by_bot = pd.DataFrame(rows)

    # Errors
    df_errors = counter_to_df(error_paths, "requests").rename(columns={"key": "path"})
    df_error_samples = pd.DataFrame(error_samples)

    # -----------------------
    # INSIGHTS
    # -----------------------
    st.header("SEO Insights (Auto)")
    insights, actions = bot_insights(df_bots, df_status, df_xhosts, df_status_by_bot, df_pathgroup_by_bot, df_errors)

    if insights:
        for i, s in enumerate(insights, 1):
            st.write(f"{i}. {s}")
    else:
        st.write("No major issues detected from this sample window.")

    if actions:
        st.subheader("Prioritized actions")
        for i, a in enumerate(actions, 1):
            st.write(f"{i}. {a}")

    # -----------------------
    # REPORTS + CHARTS
    # -----------------------
    st.header("Reports & Charts")

    # Requests by bot
    st.subheader("Requests by bot")
    st.dataframe(df_bots)
    plot_bar(df_bots.head(20), "bot", "requests", "Requests by bot (Top 20)", rotate=45)

    # Status overall
    st.subheader("Status codes (overall)")
    st.dataframe(df_status)
    plot_bar(df_status, "status", "requests", "Status codes (overall)", rotate=0)

    # Methods
    st.subheader("Methods (overall)")
    st.dataframe(df_methods)
    plot_bar(df_methods.head(15), "method", "requests", "HTTP methods (Top 15)", rotate=25)

    # Hosts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("cs(Host) distribution")
        st.dataframe(df_hosts)
        plot_bar(df_hosts.head(10), "cs(Host)", "requests", "cs(Host) distribution", rotate=25)
    with c2:
        st.subheader("x-host-header distribution")
        st.dataframe(df_xhosts)
        plot_bar(df_xhosts.head(10), "x-host-header", "requests", "x-host-header distribution", rotate=25)

    # Stacked status mix by bot
    st.subheader("Status mix by bot (stacked %)")
    if not df_status_by_bot.empty:
        st.dataframe(df_status_by_bot.sort_values(["bot", "requests"], ascending=[True, False]))
        plot_stacked_share(df_status_by_bot, "bot", "status", "requests", "Status mix by bot (Share %)", top_n=12)
    else:
        st.info("No status-by-bot data available.")

    # Stacked crawl focus by bot
    st.subheader("Crawl focus by bot (path groups, stacked %)")
    if not df_pathgroup_by_bot.empty:
        st.dataframe(df_pathgroup_by_bot.sort_values(["bot", "requests"], ascending=[True, False]))
        plot_stacked_share(df_pathgroup_by_bot, "bot", "path_group", "requests", "Crawl focus by bot (Share %)", top_n=12)
    else:
        st.info("No path-group-by-bot data available.")

    # Top URLs by bot (interactive)
    st.subheader("Top URLs by bot")
    if not df_top_urls_by_bot.empty:
        bot_choice = st.selectbox("Choose bot", df_bots["bot"].tolist(), index=0)
        show_n = st.slider("Show top N URLs", 10, 100, 30)

        top_urls = pd.DataFrame(url_counts_by_bot[bot_choice].most_common(show_n), columns=["path", "requests"])
        st.dataframe(top_urls)

        # chart
        chart_df = top_urls.copy()
        chart_df["path_short"] = chart_df["path"].astype(str).apply(lambda s: s if len(s) <= 70 else s[:67] + "...")
        plot_bar(chart_df.head(20), "path_short", "requests", f"Top URLs for {bot_choice} (Top 20)", rotate=75)
    else:
        st.info("No URL data available.")

    # Errors (table only; chart intentionally removed)
    st.subheader("4xx/5xx error paths (table)")
    st.dataframe(df_errors.head(100))
    st.subheader("Error samples")
    st.dataframe(df_error_samples)

    # -----------------------
    # DOWNLOADS (ALL CSVs)
    # -----------------------
    st.header("Download CSVs")

    downloads = {
        "requests_by_bot.csv": df_bots,
        "status_overall.csv": df_status,
        "methods_overall.csv": df_methods,
        "cs_host_distribution.csv": df_hosts,
        "x_host_header_distribution.csv": df_xhosts,
        "status_by_bot.csv": df_status_by_bot,
        "path_groups_by_bot.csv": df_pathgroup_by_bot,
        "top_urls_by_bot.csv": df_top_urls_by_bot,
        "error_paths_4xx_5xx.csv": df_errors,
        "error_samples.csv": df_error_samples,
    }

    for fname, dfo in downloads.items():
        st.download_button(
            label=f"Download {fname}",
            data=dfo.to_csv(index=False),
            file_name=fname,
            mime="text/csv"
        )

else:
    st.info("Upload files (max 5), then click **Run analysis**.")