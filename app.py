import streamlit as st
import json
import gzip
import io
import urllib.parse
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="CloudFront Bot Log Analyzer", layout="wide")
st.title("CloudFront Bot Log Analyzer")
st.markdown("*by [Islam Eldiasti](https://www.linkedin.com/in/islam-eldiasti/)*")

# -----------------------
# Bot classification
# Returns (bot_name, category)
# Categories: "AI Crawler", "Search Engine", "SEO Tool", "Social Preview", "Other Bot", "Human"
# -----------------------
def classify_bot(ua: str):
    u = (ua or "").lower()

    # --- AI / LLM Crawlers ---
    if "oai-searchbot" in u:         return "OAI-SearchBot",       "AI Crawler"
    if "gptbot" in u:                return "GPTBot",              "AI Crawler"
    if "chatgpt-user" in u:          return "ChatGPT-User",        "AI Crawler"
    if "claudebot" in u:             return "ClaudeBot",           "AI Crawler"
    if "anthropic-ai" in u:          return "Anthropic-AI",        "AI Crawler"
    if "perplexitybot" in u:         return "PerplexityBot",       "AI Crawler"
    if "gemini" in u and "bot" in u: return "GeminiBot",           "AI Crawler"
    if "meta-externalagent" in u:    return "Meta-ExternalAgent",  "AI Crawler"
    if "bytespider" in u:            return "ByteSpider",          "AI Crawler"
    if "diffbot" in u:               return "Diffbot",             "AI Crawler"
    if "cohere-ai" in u:             return "Cohere-AI",           "AI Crawler"
    if "youbot" in u:                return "YouBot",              "AI Crawler"

    # --- Ad Bots ---
    if "adsbot-google-mobile" in u:  return "AdsBot-Google-Mobile", "Ad Bot"
    if "adsbot-google" in u:         return "AdsBot-Google",         "Ad Bot"
    if "mediapartners-google" in u:  return "Mediapartners-Google",  "Ad Bot"
    if "google-adwords" in u:        return "Google-AdWords",        "Ad Bot"
    if "adidxbot" in u:              return "AdIdxBot (Bing Ads)",   "Ad Bot"
    if "bingpreview" in u:           return "BingPreview",           "Ad Bot"
    if "yandex-direct" in u:         return "Yandex-Direct",         "Ad Bot"
    if "facebookads" in u:           return "Facebook Ads",          "Ad Bot"
    if "twitterads" in u:            return "Twitter Ads",           "Ad Bot"

    # --- Search Engines ---
    if "googlebot" in u:             return "Googlebot",           "Search Engine"
    if "google-inspectiontool" in u: return "Google-Inspection",   "Search Engine"
    if "bingbot" in u:               return "Bingbot",             "Search Engine"
    if "applebot" in u:              return "Applebot",            "Search Engine"
    if "yandex" in u:                return "Yandex",              "Search Engine"
    if "duckduckbot" in u:           return "DuckDuckBot",         "Search Engine"
    if "baiduspider" in u:           return "Baiduspider",         "Search Engine"
    if "sogou" in u:                 return "Sogou",               "Search Engine"
    if "exabot" in u:                return "Exabot",              "Search Engine"
    if "ia_archiver" in u:           return "Alexa/Archive",       "Search Engine"
    if "petalbot" in u:              return "PetalBot",            "Search Engine"
    if "amazonbot" in u:             return "Amazonbot",           "Search Engine"
    if "slurp" in u:                 return "Yahoo-Slurp",         "Search Engine"

    # --- SEO Tools ---
    if "ahrefsbot" in u:             return "AhrefsBot",           "SEO Tool"
    if "semrushbot" in u:            return "SemrushBot",          "SEO Tool"
    if "mj12bot" in u:               return "MJ12bot",             "SEO Tool"
    if "majestic" in u:              return "Majestic",            "SEO Tool"
    if "dotbot" in u:                return "DotBot",              "SEO Tool"
    if "rogerbot" in u:              return "Rogerbot (Moz)",      "SEO Tool"
    if "moz.com" in u:               return "Moz",                 "SEO Tool"
    if "seokicks" in u:              return "SEOkicks",            "SEO Tool"
    if "sistrix" in u:               return "Sistrix",             "SEO Tool"
    if "screaming frog" in u:        return "Screaming Frog",      "SEO Tool"
    if "seobility" in u:             return "Seobility",           "SEO Tool"
    if "serpstat" in u:              return "Serpstat",            "SEO Tool"

    # --- Social Previews ---
    if "facebookexternalhit" in u or "facebot" in u:
                                     return "Meta",                "Social Preview"
    if "twitterbot" in u:            return "Twitterbot",          "Social Preview"
    if "linkedinbot" in u:           return "LinkedInBot",         "Social Preview"
    if "whatsapp" in u:              return "WhatsApp",            "Social Preview"
    if "telegrambot" in u:           return "TelegramBot",         "Social Preview"
    if "slackbot" in u:              return "Slackbot",            "Social Preview"
    if "discordbot" in u:            return "Discordbot",          "Social Preview"

    # --- Generic Bot Patterns ---
    if any(k in u for k in [" bot", "bot/", "crawler", "spider", "crawl"]):
        return "OtherBot", "Other Bot"

    return "Human/Browser", "Human"


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

def build_path_classifier(url_counts_by_bot: dict, top_n: int = 15):
    """
    Auto-detects the top N first-level path segments from the log data
    and returns a classifier function. Works on any website automatically.
    """
    segment_counts = Counter()
    for paths in url_counts_by_bot.values():
        for path, cnt in paths.items():
            if not path or path == "/":
                continue
            # Extract first segment: /article/foo -> "article"
            first = path.strip("/").split("/")[0]
            if first:
                segment_counts["/" + first + "/"] += cnt

    top_segments = set(seg for seg, _ in segment_counts.most_common(top_n))

    def classify(path: str) -> str:
        if not path:                  return "other"
        if path == "/":               return "/ (homepage)"
        if path == "/robots.txt":     return "/robots.txt"
        first = path.strip("/").split("/")[0]
        seg = "/" + first + "/" if first else None
        if seg and seg in top_segments:
            return seg + "*"
        return "other"

    return classify

def counter_to_df(counter: Counter, value_col="count"):
    if not counter:
        return pd.DataFrame(columns=["key", value_col, "share_%"])
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

def plot_pie(labels, sizes, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)

def plot_stacked_share(df, index_col, category_col, value_col, title, top_n=12):
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

    http301   = get_val(df_status, "status", 301, "requests")
    http304   = get_val(df_status, "status", 304, "requests")
    err_total = int(df_errors["requests"].sum()) if (df_errors is not None and not df_errors.empty) else 0
    www       = get_val(df_xhosts, "x-host-header", "www.almashhad.com", "requests")
    nonwww    = get_val(df_xhosts, "x-host-header", "almashhad.com", "requests")

    def pct(n, d): return round(n/d*100, 2) if d else 0.0

    insights, actions = [], []

    redir_rate = pct(http301, total)
    if redir_rate >= 5:
        insights.append(f"High 301 redirect rate: {redir_rate}% (301={http301}).")
        actions.append("P0: Reduce redirects by enforcing a single canonical host (https + www) and removing redirect chains.")
    elif redir_rate >= 2:
        insights.append(f"Moderate 301 redirect rate: {redir_rate}%.")
        actions.append("P1: Audit top 301 URLs and remove avoidable redirects (host, slash, encoding).")

    nonwww_rate = pct(nonwww, www + nonwww)
    if nonwww_rate >= 2:
        insights.append(f"Non-www host usage: {nonwww_rate}% (`almashhad.com`) – likely generating redirects.")
        actions.append("P0: Normalize everything to https://www.almashhad.com (sitemaps, canonicals, internal links, OG).")

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

    if err_total:
        insights.append(f"4xx/5xx rate: {pct(err_total, total)}% (count={err_total}).")
        top_err = df_errors.iloc[0]
        actions.append(f"P2: Fix top recurring error path: {top_err['path']} (hits={int(top_err['requests'])}).")

    seen = set()
    actions = [a for a in actions if not (a in seen or seen.add(a))]
    return insights, actions


# ======================
# UI
# ======================
uploaded_files = st.file_uploader(
    "Upload CloudFront log files (JSON lines, .gz or uncompressed).",
    accept_multiple_files=True
)

run = st.button("Run analysis")

# -----------------------
# STEP 1: Process files only when Run is clicked
# Store everything in session_state so filter reruns don't wipe results
# -----------------------
if run and uploaded_files:

    total_requests  = 0
    bot_counts      = Counter()
    category_counts = Counter()
    status_counts   = Counter()
    method_counts   = Counter()
    host_counts     = Counter()
    xhost_counts    = Counter()

    status_by_bot     = defaultdict(Counter)
    pathgroup_by_bot  = defaultdict(Counter)
    url_counts_by_bot = defaultdict(Counter)
    bot_to_category   = {}

    error_paths   = Counter()
    error_samples = []
    paths_404     = Counter()
    samples_404   = []

    progress    = st.progress(0)
    total_files = len(uploaded_files)

    for i, uf in enumerate(uploaded_files, start=1):
        raw_bytes = uf.read()
        if raw_bytes[:2] == b'\x1f\x8b':
            content = gzip.decompress(raw_bytes).decode("utf-8", errors="ignore")
        else:
            content = raw_bytes.decode("utf-8", errors="ignore")

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            total_requests += 1

            ua            = safe_unquote(rec.get("cs(User-Agent)", ""))
            bot, category = classify_bot(ua)
            path          = rec.get("cs-uri-stem", "") or ""
            host          = rec.get("cs(Host)", "") or ""
            xhost         = rec.get("x-host-header", "") or ""
            method        = rec.get("cs-method", "") or ""

            try:
                status = int(rec.get("sc-status", None))
            except Exception:
                status = None

            bot_counts[bot]           += 1
            category_counts[category] += 1
            host_counts[host]         += 1
            xhost_counts[xhost]       += 1
            method_counts[method]     += 1
            bot_to_category[bot]       = category

            if status is not None:
                status_counts[status]      += 1
                status_by_bot[bot][status] += 1

                if 400 <= status <= 599:
                    error_paths[path] += 1
                    if len(error_samples) < 30:
                        error_samples.append({"status": status, "path": path, "bot": bot, "ua": ua[:140]})

                if status == 404:
                    paths_404[path] += 1
                    if len(samples_404) < 50:
                        samples_404.append({"path": path, "bot": bot, "ua": ua[:140]})

            if path:
                url_counts_by_bot[bot][path] += 1

        progress.progress(i / total_files)

    # Build dynamic path classifier from actual log data
    path_group = build_path_classifier(url_counts_by_bot, top_n=15)

    # Rebuild pathgroup_by_bot using the dynamic classifier
    pathgroup_by_bot = defaultdict(Counter)
    for bot, paths in url_counts_by_bot.items():
        for path, cnt in paths.items():
            pg = path_group(path)
            pathgroup_by_bot[bot][pg] += cnt

    # Build DataFrames
    df_bots = counter_to_df(bot_counts, "requests").rename(columns={"key": "bot"})
    df_bots["category"] = df_bots["bot"].map(bot_to_category)

    df_categories = counter_to_df(category_counts, "requests").rename(columns={"key": "category"})
    df_status     = counter_to_df(status_counts, "requests").rename(columns={"key": "status"}).sort_values("status")
    df_methods    = counter_to_df(method_counts, "requests").rename(columns={"key": "method"})
    df_hosts      = counter_to_df(host_counts, "requests").rename(columns={"key": "cs(Host)"})
    df_xhosts     = counter_to_df(xhost_counts, "requests").rename(columns={"key": "x-host-header"})

    rows = []
    for b, c in status_by_bot.items():
        total_b = sum(c.values()) or 1
        for stt, cnt in c.items():
            rows.append({"bot": b, "category": bot_to_category.get(b, ""), "status": stt,
                         "requests": cnt, "share_within_bot_%": round(cnt/total_b*100, 2)})
    df_status_by_bot = pd.DataFrame(rows) if rows else pd.DataFrame()

    rows = []
    for b, c in pathgroup_by_bot.items():
        total_b = sum(c.values()) or 1
        for pg, cnt in c.items():
            rows.append({"bot": b, "category": bot_to_category.get(b, ""), "path_group": pg,
                         "requests": cnt, "share_within_bot_%": round(cnt/total_b*100, 2)})
    df_pathgroup_by_bot = pd.DataFrame(rows) if rows else pd.DataFrame()

    TOP_N = 50
    rows = []
    for b, c in url_counts_by_bot.items():
        for p, cnt in c.most_common(TOP_N):
            rows.append({"bot": b, "category": bot_to_category.get(b, ""), "path": p, "requests": cnt})
    df_top_urls_by_bot = pd.DataFrame(rows) if rows else pd.DataFrame()

    df_errors        = counter_to_df(error_paths, "requests").rename(columns={"key": "path"})
    df_error_samples = pd.DataFrame(error_samples)
    df_404_paths     = counter_to_df(paths_404, "requests").rename(columns={"key": "path"})
    df_404_samples   = pd.DataFrame(samples_404)

    # Save everything to session_state
    st.session_state["results"] = {
        "total_requests":    total_requests,
        "total_files":       total_files,
        "df_bots":           df_bots,
        "df_categories":     df_categories,
        "df_status":         df_status,
        "df_methods":        df_methods,
        "df_hosts":          df_hosts,
        "df_xhosts":         df_xhosts,
        "df_status_by_bot":  df_status_by_bot,
        "df_pathgroup_by_bot": df_pathgroup_by_bot,
        "df_top_urls_by_bot": df_top_urls_by_bot,
        "df_errors":         df_errors,
        "df_error_samples":  df_error_samples,
        "df_404_paths":      df_404_paths,
        "df_404_samples":    df_404_samples,
        "url_counts_by_bot": dict(url_counts_by_bot),
    }

# -----------------------
# STEP 2: Display results from session_state
# Runs on every rerender — including filter changes
# -----------------------
if "results" in st.session_state:
    r = st.session_state["results"]

    total_requests     = r["total_requests"]
    total_files        = r["total_files"]
    df_bots            = r["df_bots"]
    df_categories      = r["df_categories"]
    df_status          = r["df_status"]
    df_methods         = r["df_methods"]
    df_hosts           = r["df_hosts"]
    df_xhosts          = r["df_xhosts"]
    df_status_by_bot   = r["df_status_by_bot"]
    df_pathgroup_by_bot= r["df_pathgroup_by_bot"]
    df_top_urls_by_bot = r["df_top_urls_by_bot"]
    df_errors          = r["df_errors"]
    df_error_samples   = r["df_error_samples"]
    df_404_paths       = r["df_404_paths"]
    df_404_samples     = r["df_404_samples"]
    url_counts_by_bot  = r["url_counts_by_bot"]

    all_bots = sorted(df_bots["bot"].tolist())

    st.success(f"✅ Processed **{total_requests:,}** requests from **{total_files}** file(s).")

    # -----------------------
    # BOT FILTER — main page
    # -----------------------
    with st.expander("🔍 Bot Filter — click to expand", expanded=False):
        selected_bots = st.multiselect(
            "Select bots to include in all charts and tables:",
            options=all_bots,
            default=all_bots
        )
    if not selected_bots:
        st.warning("No bots selected — showing all.")
        selected_bots = all_bots

    def filt_bot(df, col="bot"):
        if df.empty or col not in df.columns:
            return df
        return df[df[col].isin(selected_bots)]

    df_bots_f             = filt_bot(df_bots)
    df_status_by_bot_f    = filt_bot(df_status_by_bot)
    df_pathgroup_by_bot_f = filt_bot(df_pathgroup_by_bot)
    df_top_urls_by_bot_f  = filt_bot(df_top_urls_by_bot)

    # -----------------------
    # OVERVIEW: Bot vs Human
    # -----------------------
    st.header("📊 Traffic Overview")
    col1, col2 = st.columns([1, 2])

    with col1:
        if not df_categories.empty:
            human_total = int(df_categories[df_categories["category"] == "Human"]["requests"].sum()) \
                          if "Human" in df_categories["category"].values else 0
            bot_total   = total_requests - human_total
            plot_pie(
                ["Human / Browser", "Bots & Crawlers"],
                [human_total, bot_total],
                "Bot vs. Human Traffic"
            )

    with col2:
        st.subheader("Traffic by Bot Category")
        st.dataframe(df_categories, use_container_width=True)
        plot_bar(df_categories, "category", "requests", "Requests by Category", rotate=25)

    # -----------------------
    # SEO INSIGHTS
    # -----------------------
    st.header("🔎 SEO Insights (Auto)")
    insights, actions = bot_insights(
        df_bots_f, df_status, df_xhosts,
        df_status_by_bot_f, df_pathgroup_by_bot_f, df_errors
    )

    if insights:
        for idx, s in enumerate(insights, 1):
            st.write(f"{idx}. {s}")
    else:
        st.write("No major issues detected from this sample window.")

    if actions:
        st.subheader("Prioritized Actions")
        for idx, a in enumerate(actions, 1):
            st.write(f"{idx}. {a}")

    # -----------------------
    # REPORTS + CHARTS
    # -----------------------
    st.header("📈 Reports & Charts")

    st.subheader("Requests by Bot")
    st.dataframe(df_bots_f, use_container_width=True)
    plot_bar(df_bots_f.head(20), "bot", "requests", "Requests by Bot (Top 20)", rotate=45)

    st.subheader("Status Codes (Overall)")
    st.dataframe(df_status, use_container_width=True)
    plot_bar(df_status, "status", "requests", "Status Codes (Overall)", rotate=0)

    st.subheader("HTTP Methods")
    st.dataframe(df_methods, use_container_width=True)
    plot_bar(df_methods.head(15), "method", "requests", "HTTP Methods (Top 15)", rotate=25)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("cs(Host) Distribution")
        st.dataframe(df_hosts, use_container_width=True)
        plot_bar(df_hosts.head(10), "cs(Host)", "requests", "cs(Host) Distribution", rotate=25)
    with col2:
        st.subheader("x-host-header Distribution")
        st.dataframe(df_xhosts, use_container_width=True)
        plot_bar(df_xhosts.head(10), "x-host-header", "requests", "x-host-header Distribution", rotate=25)

    st.subheader("Status Mix by Bot (Stacked %)")
    if not df_status_by_bot_f.empty:
        st.dataframe(df_status_by_bot_f.sort_values(["bot", "requests"], ascending=[True, False]), use_container_width=True)
        plot_stacked_share(df_status_by_bot_f, "bot", "status", "requests", "Status Mix by Bot (Share %)", top_n=12)
    else:
        st.info("No status-by-bot data available.")

    st.subheader("Crawl Focus by Bot (Path Groups, Stacked %)")
    if not df_pathgroup_by_bot_f.empty:
        st.dataframe(df_pathgroup_by_bot_f.sort_values(["bot", "requests"], ascending=[True, False]), use_container_width=True)
        plot_stacked_share(df_pathgroup_by_bot_f, "bot", "path_group", "requests", "Crawl Focus by Bot (Share %)", top_n=12)
    else:
        st.info("No path-group-by-bot data available.")

    st.subheader("Top URLs by Bot")
    if not df_top_urls_by_bot_f.empty:
        bot_choice = st.selectbox("Choose bot", df_bots_f["bot"].tolist(), index=0)
        show_n     = st.slider("Show top N URLs", 10, 100, 30)
        top_urls   = pd.DataFrame(url_counts_by_bot[bot_choice].most_common(show_n), columns=["path", "requests"])
        st.dataframe(top_urls, use_container_width=True)
        chart_df = top_urls.copy()
        chart_df["path_short"] = chart_df["path"].astype(str).apply(
            lambda s: s if len(s) <= 70 else s[:67] + "..."
        )
        plot_bar(chart_df.head(20), "path_short", "requests", f"Top URLs for {bot_choice} (Top 20)", rotate=75)
    else:
        st.info("No URL data available.")

    # -----------------------
    # 404 PATHS — dedicated
    # -----------------------
    st.header("🚫 Top 404 Paths")
    if not df_404_paths.empty:
        st.caption(f"{len(df_404_paths):,} unique paths returned 404.")
        st.dataframe(df_404_paths.head(100), use_container_width=True)
        st.subheader("404 Samples (by Bot)")
        st.dataframe(df_404_samples, use_container_width=True)
    else:
        st.success("No 404 errors found in this log set. 🎉")

    st.header("⚠️ All 4xx / 5xx Error Paths")
    st.dataframe(df_errors.head(100), use_container_width=True)

    # -----------------------
    # DOWNLOAD — single Excel
    # -----------------------
    st.header("⬇️ Download Report")

    sheets = {
        "Summary by Category":  df_categories,
        "Requests by Bot":      df_bots,
        "Status Overall":       df_status,
        "Methods Overall":      df_methods,
        "CS Host Distribution": df_hosts,
        "X-Host Distribution":  df_xhosts,
        "Status by Bot":        df_status_by_bot,
        "Path Groups by Bot":   df_pathgroup_by_bot,
        "Top URLs by Bot":      df_top_urls_by_bot,
        "404 Paths":            df_404_paths,
        "404 Samples":          df_404_samples,
        "Error Paths 4xx5xx":   df_errors,
        "Error Samples":        df_error_samples,
    }

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, dfo in sheets.items():
            dfo.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)

    st.download_button(
        label="⬇️ Download Full Report (.xlsx)",
        data=buffer,
        file_name="cloudfront_bot_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Upload one or more CloudFront log files (.gz or uncompressed), then click **Run analysis**.")

    # -----------------------
    # MOCK DATA PREVIEW
    # -----------------------
    st.markdown("---")
    st.markdown("### 👇 Sample Output Preview")
    st.caption("This is what your analysis will look like — populated with demo data.")

    import numpy as np
    rng = np.random.default_rng(42)

    # Mock bots
    mock_bots = ["Googlebot", "AhrefsBot", "SemrushBot", "GPTBot", "ClaudeBot",
                 "PerplexityBot", "Bingbot", "Human/Browser", "AdsBot-Google",
                 "Meta", "Twitterbot", "OtherBot"]
    mock_bot_reqs = [18400, 9200, 7800, 5400, 3200, 2900, 4100, 42000, 1200, 800, 600, 1100]
    total_mock = sum(mock_bot_reqs)
    df_mock_bots = pd.DataFrame({
        "bot": mock_bots,
        "requests": mock_bot_reqs,
        "share_%": [round(r/total_mock*100, 2) for r in mock_bot_reqs]
    }).sort_values("requests", ascending=False).reset_index(drop=True)

    # Mock categories
    mock_cats = ["Human", "Search Engine", "SEO Tool", "AI Crawler", "Ad Bot", "Social Preview", "Other Bot"]
    mock_cat_reqs = [42000, 22500, 17000, 11500, 1200, 1400, 1100]
    total_cat = sum(mock_cat_reqs)
    df_mock_cats = pd.DataFrame({
        "category": mock_cats,
        "requests": mock_cat_reqs,
        "share_%": [round(r/total_cat*100, 2) for r in mock_cat_reqs]
    })

    # Mock status codes
    df_mock_status = pd.DataFrame({
        "status": [200, 301, 304, 404, 403, 500],
        "requests": [78000, 8200, 5400, 3100, 420, 180],
        "share_%": [81.5, 8.6, 5.6, 3.2, 0.7, 0.4]
    })

    # Mock path groups
    mock_path_groups = ["/article/*", "/_next/*", "/tags/*", "/section/*",
                        "/api/*", "/ (homepage)", "/images/*", "/robots.txt", "other"]
    mock_pg_reqs = [38000, 18000, 12000, 9000, 7000, 4500, 3200, 1800, 2500]
    df_mock_pathgroups = pd.DataFrame({
        "path_group": mock_path_groups,
        "requests": mock_pg_reqs
    })

    # Mock 404 paths
    df_mock_404 = pd.DataFrame({
        "path": ["/article/old-story", "/tags/deleted-tag", "/section/removed",
                 "/article/broken-link", "/old-page"],
        "requests": [420, 310, 280, 195, 140],
        "share_%": [13.5, 10.0, 9.0, 6.3, 4.5]
    })

    # --- Render mock overview ---
    st.header("📊 Traffic Overview  *(sample)*")
    col1, col2 = st.columns([1, 2])

    with col1:
        human_m = 42000
        bot_m   = total_mock - human_m
        plot_pie(["Human / Browser", "Bots & Crawlers"], [human_m, bot_m], "Bot vs. Human Traffic")

    with col2:
        st.subheader("Traffic by Bot Category")
        st.dataframe(df_mock_cats, use_container_width=True)
        plot_bar(df_mock_cats, "category", "requests", "Requests by Category", rotate=25)

    # --- Mock reports ---
    st.header("📈 Reports & Charts  *(sample)*")

    st.subheader("Requests by Bot")
    st.dataframe(df_mock_bots, use_container_width=True)
    plot_bar(df_mock_bots, "bot", "requests", "Requests by Bot", rotate=45)

    st.subheader("Status Codes (Overall)")
    st.dataframe(df_mock_status, use_container_width=True)
    plot_bar(df_mock_status, "status", "requests", "Status Codes (Overall)", rotate=0)

    st.subheader("Crawl Focus by Bot (Path Groups)")
    st.dataframe(df_mock_pathgroups, use_container_width=True)

    st.header("🚫 Top 404 Paths  *(sample)*")
    st.dataframe(df_mock_404, use_container_width=True)
