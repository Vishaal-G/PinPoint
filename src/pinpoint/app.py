"""The demo UI: pick a repo, paste an issue (or give its number), see where to look.

    uv run streamlit run src/pinpoint/app.py          # http://localhost:8501

Third front-end on the same core as api.py and bot.py: live.checkout + live.report. Streamlit reruns
this whole script on every interaction, so the expensive parts are cached:
  - the embedder and LLM client: one per process (providers.py, shared with the API);
  - each report: st.cache_data keyed on (repo, commit, issue text, k). The commit, not the ref, is
    in the key, so "main" moving on gives a fresh report. Reports degraded by an LLM outage are evicted.
"""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from pinpoint import live, providers
from pinpoint.bot import link, linkify

ISSUE_URL = r"^https://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)"


def indexed_repos(root: Path = live.LIVE_DIR / "repos") -> list[str]:
    """Repos already cloned by the live path: fast to triage again (their bodies are embedded)."""
    return sorted(p.name.replace("__", "/", 1) for p in root.iterdir() if p.is_dir()) if root.exists() else []


# The leading underscore tells st.cache_data not to hash that argument: the embedder and LLM client
# aren't hashable data, and they don't change the answer's identity. The commit sha does.
@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=100)
def cached_report(repo: str, sha: str, text: str, k: int, _snapshot: Path) -> dict:
    return live.report(_snapshot, sha, repo, text, providers.get_embedder(), providers.get_llm(), k=k)


def run(repo: str, ref: str | None, text: str, k: int) -> dict:
    from git import GitCommandError

    t0 = time.perf_counter()
    with st.status(f"Triaging {repo}...", expanded=True) as status:
        with providers.repo_lock(repo):  # one checkout per repo, shared by every session
            st.write(f"Checking out {repo} at {ref or 'the default branch'}...")
            try:
                snapshot, sha = live.checkout(repo, ref)
            except GitCommandError:
                status.update(label="Checkout failed", state="error")
                raise RuntimeError(f"cannot check out {repo} at {ref or 'the default branch'}") from None
            st.write(f"Ranking functions at {sha[:7]} (a repo seen for the first time embeds every "
                     "function once: minutes, not seconds)...")
            try:
                rep = cached_report(repo, sha, text, k, snapshot)
            except Exception as e:  # embedding endpoint down: no ranking at all
                status.update(label="Embedding failed", state="error")
                raise RuntimeError(f"embedding failed: {e}") from None
        if rep["source"] != "agent" or rep["explanation"] is None:  # LLM was down: retry next time
            cached_report.clear(repo, sha, text, k, snapshot)
        status.update(label=f"Done in {time.perf_counter() - t0:.0f} s", state="complete", expanded=False)
    return rep


def show(rep: dict) -> None:
    repo, sha = rep["repo"], rep["commit"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ranked by", "agent" if rep["source"] == "agent" else "dense only")
    c2.metric("Commit", sha[:7])
    c3.metric("LLM tokens", f"{rep['prompt_tokens'] + rep['completion_tokens']:,}")
    for w in rep["warnings"]:
        st.warning(w)

    st.subheader("Where to look")
    st.dataframe(pd.DataFrame([{
        "#": i, "function": s["function_id"].split("::", 1)[1], "file": s["path"],
        "lines": f"{s['start_line']}-{s['end_line']}",
        "owner": f"{s['owner']['name']} ({s['owner']['share']:.0%})" if s["owner"] else "unknown",
        "code": link(repo, sha, s["path"], s["start_line"], s["end_line"])}
        for i, s in enumerate(rep["suspects"], 1)]),
        hide_index=True, width="stretch",
        column_config={"code": st.column_config.LinkColumn("code", display_text="open")})

    if rep["explanation"]:
        st.subheader("Why")
        st.markdown(linkify(rep["explanation"], repo, sha))
        cites = rep["explanation"]["citations"]
        if cites:
            ok = sum(c["valid"] for c in cites)
            st.caption(f"{ok}/{len(cites)} citations point inside the code the model was shown.")
    with st.expander("Raw report"):
        st.json(rep)


def main() -> None:
    st.set_page_config(page_title="Pinpoint", page_icon="📍", layout="wide")
    st.title("📍 Pinpoint")
    st.caption("Paste a bug report; get the functions most likely responsible, who owns them, and why.")

    with st.form("triage"):
        repo = st.selectbox("Repository (owner/name)", indexed_repos(), index=None, accept_new_options=True,
                            placeholder="pick an indexed repo or type owner/name", key="repo")
        ref = st.text_input("Commit or branch (optional)", placeholder="default branch", key="ref")
        url = st.text_input("GitHub issue URL (optional, instead of pasting)",
                            placeholder="https://github.com/owner/name/issues/123", key="url")
        title = st.text_input("Issue title", key="title")
        body = st.text_area("Issue body", height=200, key="body")
        k = st.slider("Suspects to show", 1, 10, 5)
        submitted = st.form_submit_button("Triage", type="primary")

    if submitted:
        import re

        if m := re.match(ISSUE_URL, url.strip()):
            repo = repo or m[1]
            try:
                text = live.github_issue(m[1], int(m[2]))
            except Exception as e:
                st.error(f"could not fetch {url}: {e}")
                return
        else:
            text = f"{title}\n{body}".strip()
        if not repo or not re.match(live.REPO_PATTERN, repo):
            st.error("Repository must look like owner/name.")
            return
        if not text:
            st.error("Give an issue title/body or a GitHub issue URL.")
            return
        try:
            st.session_state.report = run(repo, ref.strip() or None, text, k)
        except RuntimeError as e:
            st.session_state.pop("report", None)
            st.error(str(e))
            return

    if "report" in st.session_state:  # survives reruns from other widgets
        show(st.session_state.report)


main()
