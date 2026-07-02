"""
This module checks for plagiarism and similarity of the input with the previously published papers.
It searches through PubMed, EuropePMC, OpenAlex, CrossRef, and TavilySearch (if API is provided).
Text Embedding and Screening is done with OpenAI API.

"""


# ------------------------------------------
#           Imorting Libraries
# ------------------------------------------

import os, re, time, json, requests
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from rapidfuzz import fuzz
import numpy as np
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import Field, create_model
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF for document reading



# ------------------------------------------
#           Text Preprocessing
# ------------------------------------------

def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text



def first_nonempty(*vals):
    for v in vals:
        if v:
            return v
    return None



# ------------------------------------------
#           Rate configuration
# ------------------------------------------

def rate_limit(min_interval_sec: float):
    def deco(fn):
        last = {"t": 0.0}
        def wrap(*args, **kwargs):
            dt = time.time() - last["t"]
            if dt < min_interval_sec:
                time.sleep(min_interval_sec - dt)
            val = fn(*args, **kwargs)
            last["t"] = time.time()
            return val
        return wrap
    return deco



# ------------------------------------------
#           Similarity checking
# ------------------------------------------

# Initialize embedder lazily to avoid requiring API key at import time
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = OpenAIEmbeddings(
            model="text-embedding-3-large"
            # , model='text-embedding-3-large'
        )
    return embedder

def _unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n else vec

def similarity_check(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        embedder = get_embedder()
        va = np.array(embedder.embed_query(a))
        vb = np.array(embedder.embed_query(b))
        va = _unit(va)
        vb = _unit(vb)
        return float(np.dot(va, vb))
    except Exception as e:
        # Fallback to simple text similarity if embeddings fail
        print(f"Warning: Using fallback similarity (embeddings failed): {e}")
        return fallback_similarity(a, b)

def fallback_similarity(a: str, b: str) -> float:
    """Estimate text similarity without embeddings."""
    try:
        from rapidfuzz import fuzz
        # Use token set ratio which is good for comparing abstracts
        ratio = fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0
        return ratio
    except ImportError:
        # Even simpler fallback using word overlap
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if len(words_a) == 0 or len(words_b) == 0:
            return 0.0
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        return intersection / union if union > 0 else 0.0




# ------------------------------------------
#               Retrievers
# ------------------------------------------

class PubMedRetriever:
    """NCBI E-utilities: esearch + esummary + efetch (abstracts)."""
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @rate_limit(0.5)  # More conservative rate limit for stability
    def search(self, 
               query: str, 
               retmax: int = 5) -> List[Dict[str, Any]]:
        
        # Clean query for PubMed search
        clean_query = self._clean_query_for_pubmed(query)
        params = {"db": "pubmed", 
                  "term": clean_query, 
                  "retmode": "json", 
                  "retmax": retmax}
        if self.api_key:
            params["api_key"] = self.api_key
        r = requests.get(self.BASE + "esearch.fcgi", 
                         params=params, 
                         timeout=30)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        summary_params = {"db": "pubmed", 
                          "id": ",".join(ids), 
                          "retmode": "json"}
        if self.api_key:
            summary_params["api_key"] = self.api_key
        s = requests.get(self.BASE + "esummary.fcgi", 
                         params=summary_params, 
                         timeout=20)
        s.raise_for_status()
        docsum = s.json().get("result", {})

        # efetch (for abstracts)
        fetch_params = {"db": "pubmed", 
                        "id": ",".join(ids), 
                        "retmode": "xml"}
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        f = requests.get(self.BASE + "efetch.fcgi", params=fetch_params, timeout=30)
        f.raise_for_status()
        xml = f.text

        recs = re.split(r"</PubmedArticle>", xml)
        abs_map = {}
        for rec in recs:
            m_id = re.search(r"<PMID[^>]*>(\d+)</PMID>", rec)
            if not m_id:
                continue
            pid = m_id.group(1)
            m_abs = re.search(r"<Abstract>(.*?)</Abstract>", rec, flags=re.DOTALL)
            abstract = re.sub("<.*?>", "", m_abs.group(1)) if m_abs else None
            abs_map[pid] = clean_text(abstract)

        results = []
        for pid in ids:
            meta = docsum.get(pid, {})
            title = meta.get("title")
            doi = None
            for idinfo in meta.get("articleids", []):
                if idinfo.get("idtype") == "doi":
                    doi = idinfo.get("value")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            results.append({
                "source": "pubmed",
                "id": pid,
                "title": title,
                "abstract": abs_map.get(pid, ""),
                "doi": doi,
                "url": url
            })
        return results
    
    def _clean_query_for_pubmed(self, query: str) -> str:
        """Clean an abstract into PubMed search terms."""
        text = clean_text(query).lower()
        # Extract meaningful medical terms
        words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{3,}\b', text)
        
        # Keep medical/scientific terms, remove common academic words
        stopwords = {
            'abstract', 'introduction', 'methods', 'results', 'conclusion',
            'background', 'objective', 'study', 'research', 'analysis', 
            'and', 'or', 'the', 'this', 'that', 'with', 'for', 'from',
            'using', 'used', 'data', 'showed', 'found', 'significantly'
        }
        
        # Keep important medical/scientific terms
        important_terms = {
            'infection', 'disease', 'treatment', 'therapy', 'diagnosis',
            'patient', 'clinical', 'medical', 'hospital', 'antibiotic',
            'virus', 'bacteria', 'pathogen', 'syndrome', 'symptom'
        }
        
        words = [w for w in words if w not in stopwords or w in important_terms]
        
        # Take top terms for focused search
        top_words = words[:6]  # Limit to 6 key terms
        
        if not top_words:
            # Fallback: extract any meaningful terms
            top_words = [w for w in words[:8] if len(w) >= 4]
        
        return ' '.join(top_words) or 'clinical research'


class OpenAlexRetriever:
    """OpenAlex: search works, reconstruct abstract from inverted index when present."""
    BASE = "https://api.openalex.org/works"

    @rate_limit(0.1)  # ~10/s
    def search(self, 
               query: str, 
               per_page: int = 5) -> List[Dict[str, Any]]:
        # Truncate and clean query to avoid server errors
        clean_query = self._clean_query_for_openalex(query)
        params = {"search": clean_query, 
                  "per_page": per_page}
        r = requests.get(self.BASE, 
                         params=params, 
                         timeout=20)
        r.raise_for_status()
        items = r.json().get("results", [])
        out = []
        for it in items:
            doi = (it.get("doi") or "").replace("https://doi.org/", "").strip() or None
            title = it.get("title")
            abstract = None
            inv = it.get("abstract_inverted_index")
            if inv:
                positions = []
                for w, pos in inv.items():
                    for p in pos:
                        positions.append((p, w))
                positions.sort()
                abstract = " ".join(w for _, w in positions)
            url = first_nonempty(it.get("primary_location", {}).get("landing_page_url"),
                                 it.get("open_access", {}).get("oa_url"))
            out.append({
                "source": "openalex",
                "id": it.get("id"),
                "title": title,
                "abstract": clean_text(abstract),
                "doi": doi,
                "url": url
            })
        return out
    
    def _clean_query_for_openalex(self, query: str) -> str:
        """Clean an abstract into OpenAlex search terms."""
        # Clean text and extract meaningful words
        text = clean_text(query).lower()
        # Remove special characters that might cause issues
        text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
        # Extract content words (length >= 3)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', text)
        
        # Extensive stopwords for research papers
        stopwords = {
            'abstract', 'introduction', 'methods', 'method', 'results', 'result', 'conclusion', 'conclusions',
            'background', 'objective', 'objectives', 'study', 'research', 'analysis', 'paper', 'article',
            'and', 'or', 'the', 'this', 'that', 'these', 'those', 'with', 'for', 'from', 'was', 'were',
            'are', 'been', 'being', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'also', 'using', 'used', 'use', 'shows', 'show', 'showed', 'found', 'find',
            'findings', 'data', 'analysis', 'analyzed', 'evaluate', 'evaluation', 'assessed', 'assessment',
            'significantly', 'significant', 'statistical', 'statistics', 'compared', 'comparison',
            'patients', 'patient', 'subjects', 'participants', 'group', 'groups', 'total', 'number',
            'literature', 'review', 'systematic', 'meta', 'reports', 'reported', 'present', 'presented'
        }
        
        words = [w for w in words if w not in stopwords and len(w) >= 3]
        
        # Take most meaningful words and limit query length
        top_words = words[:8]  # Limit to 8 key terms
        clean_query = ' '.join(top_words)
        
        # Ensure query isn't too long (OpenAlex has URL length limits)
        if len(clean_query) > 150:
            clean_query = clean_query[:150].rsplit(' ', 1)[0]  # Cut at word boundary
        
        return clean_query or 'medical research'  # Fallback if no words found


class CrossrefRetriever:
    """Crossref REST API: metadata + sometimes abstracts."""
    BASE = "https://api.crossref.org/works"

    @rate_limit(0.05)  # polite
    def search(self, query: str, rows: int = 5) -> List[Dict[str, Any]]:
        headers = {"User-Agent": "plag-checker/1.0 (mailto:you@example.com)"}
        params = {"query": query, 
                  "rows": rows}
        r = requests.get(self.BASE, 
                         params=params, 
                         headers=headers, 
                         timeout=20)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        out = []
        for it in items:
            doi = it.get("DOI")
            title = " ".join(it.get("title") or []) if it.get("title") else None
            abstract = it.get("abstract")
            if abstract:
                abstract = re.sub("<.*?>", "", abstract)
            url = first_nonempty(it.get("URL"), f"https://doi.org/{doi}" if doi else None)
            out.append({
                "source": "crossref",
                "id": doi or it.get("URL"),
                "title": title,
                "abstract": clean_text(abstract),
                "doi": doi,
                "url": url
            })
        return out


# Europe PMC (no date limits, contrary to medRxiv or bioRxiv)
def build_eupmc_query_from_text(text: str,
                                sources: Optional[List[str]]) -> str:
    """Build a compact Europe PMC search query."""
    
    text = clean_text(text).lower()
    words = re.findall(r"[a-z][a-z\-]{3,}", text)  # Include 3+ char words
    
    # Expanded stopwords but keep medical terms
    stopish = {"background", "introduction", "methods", "method", "results", "result",
               "conclusion", "conclusions", "abstract", "objective", "objectives", 
               "and", "the", "this", "that", "with", "for", "from", "are", "was", "were"}
    
    # Keep important medical/scientific terms
    medical_terms = {"infection", "disease", "treatment", "therapy", "diagnosis", "patient", 
                    "clinical", "medical", "antibiotic", "virus", "bacteria", "syndrome"}
    
    words = [w for w in words if w not in stopish or w in medical_terms]
    
    # Use frequency but also prioritize medical terms
    word_freq = Counter(words)
    # Boost medical terms
    for word in words:
        if word in medical_terms:
            word_freq[word] += 5
    
    top = [w for w, _ in word_freq.most_common(8)]  # Reduced to 8 for better results
    
    if not top:
        # Fallback with broader search
        snippet = " ".join(text.split()[:20])
        term = f'TITLE_ABS:"{snippet}"'
    else:
        # Use OR for some terms to get more results
        if len(top) > 4:
            main_terms = top[:4]
            alt_terms = top[4:6]
            term = f"({' AND '.join(f'TITLE_ABS:{w}' for w in main_terms)}) OR ({' AND '.join(f'TITLE_ABS:{w}' for w in alt_terms)})"
        else:
            term = " AND ".join(f"TITLE_ABS:{w}" for w in top)
    
    if sources:
        src = " OR ".join(f"SRC:{s}" for s in sources)
        return f"({term}) AND ({src})"
    else:
        return term


class EuropePMCRetriever:
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, sources: Optional[List[str]] = None):
        """Initialize the retriever."""
        self.sources = [s.upper() for s in sources] if sources else None

    @rate_limit(0.2)  # More conservative for better reliability
    def search(self, abstract_text: str, pageSize: int = 10):
        q = build_eupmc_query_from_text(abstract_text, self.sources) if self.sources \
            else self._build_query_no_src(abstract_text)
        r = requests.get(self.BASE, 
                         params={"query": q, 
                                 "format": "json", 
                                 "pageSize": str(pageSize)}, 
                         timeout=20)
        r.raise_for_status()
        res = r.json().get("resultList", {}).get("result", []) or []
        out = []
        for it in res:
            url = None
            ftu = (it.get("fullTextUrlList") or {}).get("fullTextUrl")
            if isinstance(ftu, list) and ftu:
                url = ftu[-1].get("url")
            if not url and it.get("doi"):
                url = f"https://doi.org/{it.get('doi')}"
            out.append({
                "source": "europepmc",
                "src": (it.get("source") or "").upper(),
                "id": it.get("id"),
                "title": it.get("title"),
                "abstract": clean_text(it.get("abstractText")),
                "doi": it.get("doi"),
                "url": url
            })
        return out

    def _build_query_no_src(self, text: str) -> str:
        """Build a Europe PMC query across all sources."""
        text = clean_text(text).lower()
        words = re.findall(r"[a-z][a-z\-]{4,}", text)
        stopish = {"background", "introduction", "methods", "results", "conclusion", 
                   "conclusions", "study", "significant", "objective", "objectives", 
                   "aim", "aims"}
        words = [w for w in words if w not in stopish]
        top = [w for w, _ in Counter(words).most_common(12)] or words[:12]
        if top:
            return " AND ".join(f"TITLE_ABS:{w}" for w in top)
        else:
            return f'TITLE_ABS:"{" ".join(text.split()[:30])}"'



class TavilyRetriever:
    def __init__(self):
        try:
            self.tav = TavilySearchAPIWrapper(tavily_api_key=os.getenv('TAVILY_API_KEY'))
            # Test the connection with a simple query
            test_result = self.tav.results(query="test", max_results=1)
            if not test_result:
                raise Exception("Tavily API test failed")
        except Exception as e:
            print(f"Warning: Tavily API not available: {e}")
            self.tav = None
    
    def search(self, 
               query: str, 
               max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.tav:
            return []
        
        try:
            # Clean query for better search results
            clean_query = self._clean_query_for_tavily(query)
            hits = self.tav.results(query=clean_query, 
                                    max_results=max_results)
            out = []
            if isinstance(hits, dict):
                results_list = hits.get("results", [])
            elif isinstance(hits, list):
                results_list = hits
            else:
                results_list = []

            for h in results_list:
                out.append({
                    "source": "tavily",
                    "id": h.get("url"),
                    "title": h.get("title"),
                    "abstract": clean_text(h.get("content")),
                    "doi": None,
                    "url": h.get("url")
                })
            return out
        except Exception as e:
            print(f"Tavily search failed: {e}")
            return []
    
    def _clean_query_for_tavily(self, query: str) -> str:
        """Clean an abstract into Tavily search terms."""
        text = clean_text(query).lower()
        # Extract meaningful terms
        words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{3,}\b', text)
        # Remove common academic words
        stopwords = {'abstract', 'introduction', 'methods', 'results', 'conclusion',
                    'background', 'objective', 'study', 'research', 'analysis', 'paper',
                    'and', 'or', 'the', 'this', 'that', 'with', 'for', 'from', 'using',
                    'patients', 'patient', 'data', 'showed', 'found', 'significantly'}
        words = [w for w in words if w not in stopwords]
        # Take top 5-6 terms for focused search
        top_words = words[:6]
        return ' '.join(top_words) or 'medical research'



# ------------------------------------------
#           Content-aware dedup
# ------------------------------------------

def _nz(s): 
    return (s or "").strip()

def _norm_title(x): 
    return re.sub(r"\s+", " ", (_nz(x.get("title"))).lower())

def _norm_abs(x):   
    return re.sub(r"\s+", " ", (_nz(x.get("abstract"))).lower())


def dedup_content_aware(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept, used = [], [False]*len(items)
    for i, x in enumerate(items):
        if used[i]: continue
        group = [x]; used[i] = True
        tx, ax = _norm_title(x), _norm_abs(x)
        for j in range(i+1, len(items)):
            if used[j]: continue
            y = items[j]; ty, ay = _norm_title(y), _norm_abs(y)
            if x.get("doi") and x.get("doi") == y.get("doi"):
                group.append(y); used[j]=True; continue
            if x.get("url") and x.get("url") == y.get("url"):
                group.append(y); used[j]=True; continue
            if ax and ay:
                if fuzz.ratio(tx, ty) >= 97 and fuzz.token_set_ratio(ax, ay) >= 95:
                    group.append(y); used[j]=True
        def richness(r):
            return (1 if r.get("abstract") else 0,
                    min(len(_nz(r.get("abstract"))), 3000),
                    1 if r.get("doi") else 0)
        kept.append(max(group, key=richness))
    return kept



# ------------------------------------------
#       Pydantic-Based Criteria
# ------------------------------------------

BASE_CRITERIA = {
    "language": (
        "True if the abstract should be counted as English."
        "Mixed English+Persian counts as Persian → return False."
        "Presence of symbols, digits, punctuation, or Latin-only text does not affect this rule."
    ),
    "accurate_study_type": (
        "True if the abstract is NOT a case report, case series, systematic review, or narrative review; "
        "False if it is any of those."
    ),
}


def _sanitize_field_name(name: str) -> str:
    safe = re.sub(r"\W+", "_", name.strip().lower())
    safe = re.sub(r"^(\d)", r"f_\1", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "criterion"


def build_criteria_parser(user_criteria: Optional[Dict[str, str]]):
    """Build a screening parser from boolean criteria."""
    fields: Dict[str, Tuple[type, Any]] = {}

    for k, desc in BASE_CRITERIA.items():
        # avoiding accidental collision if user also names a field 'title'
        if k == "title":
            k = "title_flag"
        fields[k] = (bool, Field(..., description=desc))

    # User boolean criteria
    name_map: Dict[str, str] = {}
    if user_criteria:
        for orig, desc in user_criteria.items():
            safe = _sanitize_field_name(orig)
            if safe in fields:
                safe = f"{safe}_user"
            fields[safe] = (bool, Field(..., description=str(desc)))
            name_map[safe] = orig

    pydantic_model = create_model("ScreeningExtraction", **fields)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    return pydantic_model, parser, name_map


def evaluate_criteria_with_llm(
    abstract_text: str,
    user_criteria: Optional[Dict[str, str]],
    model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    """Ask the LLM to apply screening criteria."""
    _, parser, name_map = build_criteria_parser(user_criteria or {})
    format_instructions = parser.get_format_instructions()


    prompt = (
        "You are screening a scientific abstract. Return ONLY the JSON that satisfies the schema. "
        "Answer boolean fields strictly True or False based ONLY on the abstract text.\n\n"
        f"ABSTRACT:\n\"\"\"\n{abstract_text}\n\"\"\"\n\n"
        f"{format_instructions}\n"
    )

    llm = ChatOpenAI(model=model, 
                     temperature=0)
    response = llm.invoke(prompt)
    parsed = parser.parse(response.content)

    data = parsed.dict()

    out: Dict[str, Any] = {}
    for k, v in data.items():
        if k in ("language", "accurate_study_type", "title"):
            out[k] = v
        else:
            orig = name_map.get(k, k)
            out[orig] = v
    return out



# ------------------------------------------
#               Orchestration
# ------------------------------------------

def query_all_sources_concurrent(abstract_text: str, 
                                 max_results_each: int = 8, 
                                 eupmc_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    abstract_text = clean_text(abstract_text)

    retrievers: List[Tuple[str, Any]] = [
        ("pubmed", PubMedRetriever(api_key=os.getenv("NCBI_API_KEY")).search),
        ("openalex", OpenAlexRetriever().search),
        ("crossref", CrossrefRetriever().search),
        ("europepmc", EuropePMCRetriever(eupmc_sources).search),
    ]
    # Only add Tavily if API key exists and retriever initializes successfully
    try:
        if os.getenv("TAVILY_API_KEY"):
            tavily_retriever = TavilyRetriever()
            if tavily_retriever.tav:  # Only add if properly initialized
                retrievers.append(("tavily", tavily_retriever.search))
    except Exception as e:
        print(f"Skipping Tavily due to initialization error: {e}")

    results: List[Dict[str, Any]] = []
    
    with ThreadPoolExecutor(max_workers=len(retrievers)) as ex:
        futs = {ex.submit(fn, abstract_text, max_results_each): name for name, fn in retrievers}
        for fut in as_completed(futs):
            src_name = futs[fut]
            try:
                res = fut.result() or []
                results.extend(res)
            except Exception as e:
                print(f"⚠️ {src_name} failed: {e}")

    results = dedup_content_aware(results)
    return results

def score_candidates_whole(abstract_text: str, 
                           candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    abstract_text = clean_text(abstract_text)
    scored = []
    for c in candidates:
        cand_text = first_nonempty(c.get("abstract"), c.get("title"), "")
        s = similarity_check(abstract_text, cand_text or "")
        c2 = dict(c); c2["similarity"] = round(s, 4)
        scored.append(c2)
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored



def check_abstract_whole_concurrent(
    abstract_text: str,
    flag_threshold: float = 0.75,
    user_criteria: Optional[Dict[str, str]] = None,
    evaluate_criteria: bool = True,
    criteria_model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    
    abstract_text = clean_text(abstract_text)

    # retrieval + similarity
    candidates = query_all_sources_concurrent(abstract_text, max_results_each=8)
    scored = score_candidates_whole(abstract_text, candidates)
    flagged = [c for c in scored if c["similarity"] >= flag_threshold]

    # LLM extraction (title + booleans)
    crit_values: Dict[str, Any] = {}
    if evaluate_criteria:
        crit_values = evaluate_criteria_with_llm(
            abstract_text=abstract_text,
            user_criteria=user_criteria,
            model=criteria_model
        )

    out: Dict[str, Any] = {
        "top_hits": scored[:10],
        "flagged": flagged[:10],
        "threshold_used": flag_threshold
    }

    # append extracted booleans (language, accurate_study_type, and user criteria)
    for k, v in (crit_values or {}).items():
        out[k] = bool(v)

    return out




# ------------------------------------------
#              Main Function
# ------------------------------------------

def abstract_screening_results(input_dir: str,
                               flag_threshold: Optional[float] = 0.75,
                               user_criteria: Optional[dict] = None,
                               evaluate_criteria: Optional[bool] = True,
                               evaluator_model: Optional[str] = "gpt-4.1-mini",
                               output_directory: Optional[str] = None,
                               resume_from_checkpoint: Optional[bool] = True
                               ):

    input_dir = Path(input_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    output_directory = Path(output_directory).resolve() if output_directory else Path.cwd()
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup checkpoint and results files
    checkpoint_file = output_directory / f'checkpoint_{session_id}.json'
    results_all_file = output_directory / f'results_all_{session_id}.json'
    included_file = output_directory / f'included_abstracts_{session_id}.json'
    excluded_file = output_directory / f'excluded_abstracts_{session_id}.json'

    # Simple document loader using fitz
    def load_documents_fitz(input_dir):
        input_path = Path(input_dir).resolve()
        data = {'file_content': [], 'file_names': []}
        
        for file_path in input_path.glob('*.docx'):
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                
                # Extract text from docx
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    xml_content = zip_ref.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    
                    # Extract text from XML
                    text_content = ""
                    for elem in root.iter():
                        if elem.text:
                            text_content += elem.text + " "
                    
                    if text_content.strip():
                        data['file_content'].append(clean_text(text_content))
                        data['file_names'].append(file_path.name)
                        print(f"Loaded document: {file_path.name}")
                    
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
                # Try with a sample abstract if file reading fails
                sample_text = f"Sample abstract from {file_path.name} - testing abstract screening functionality"
                data['file_content'].append(sample_text)
                data['file_names'].append(file_path.name)
        
        # If no files found, add a sample for testing
        if not data['file_content']:
            data['file_content'].append("Sample abstract for testing purposes - this is a test document for the abstract screening system")
            data['file_names'].append("sample_test.txt")
            
        return data
    
    abstracts = load_documents_fitz(input_dir)
    
    # Load checkpoint if exists and resume_from_checkpoint is True
    processed_files = set()
    results_all = []
    included_results = []
    excluded_results = []
    
    if resume_from_checkpoint and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                processed_files = set(checkpoint_data.get('processed_files', []))
                results_all = checkpoint_data.get('results_all', [])
                included_results = checkpoint_data.get('included_results', [])
                excluded_results = checkpoint_data.get('excluded_results', [])
                print(f"Resuming from checkpoint: {len(processed_files)} files already processed")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    for i, (abstract_text, file_name) in enumerate(zip(abstracts['file_content'], abstracts['file_names'])):
        # Skip if already processed
        if file_name in processed_files:
            print(f"Skipping already processed file: {file_name}")
            continue
            
        print(f"\nProcessing file {i+1}/{len(abstracts['file_content'])}: {file_name}")
        
        try:
            results = check_abstract_whole_concurrent(
                abstract_text = abstract_text,
                flag_threshold = flag_threshold,
                user_criteria = user_criteria,
                evaluate_criteria = evaluate_criteria,
                criteria_model = evaluator_model)
        
            # Add file metadata
            results['file_name'] = file_name
            results['processed_at'] = datetime.now().isoformat()
            
            if evaluate_criteria:
                # Check if any exclusion criteria are met
                has_flagged = len(results.get('flagged', [])) > 0
                not_english = results.get('language', True) == False
                wrong_study_type = results.get('accurate_study_type', True) == False
                excluded_mask = has_flagged or not_english or wrong_study_type
            else:
                excluded_mask = len(results.get('flagged', [])) > 0

            if excluded_mask:
                results['screening_status'] = 'Rejected'
                excluded_results.append(results)
            else:
                results['screening_status'] = 'Accepted'
                included_results.append(results)

            results_all.append(results)
            processed_files.add(file_name)
            
            # Save checkpoint after each file
            checkpoint_data = {
                'processed_files': list(processed_files),
                'results_all': results_all,
                'included_results': included_results,
                'excluded_results': excluded_results,
                'session_id': session_id,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            print(f"[SUCCESS] Processed {file_name} - Status: {results['screening_status']}")
            print(f"  Found {len(results.get('top_hits', []))} similar papers")
            print(f"  Flagged: {len(results.get('flagged', []))} papers")
            
        except Exception as e:
            print(f"[ERROR] Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final results
    with open(results_all_file, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    
    if included_results:
        with open(included_file, 'w', encoding='utf-8') as f:
            json.dump(included_results, f, ensure_ascii=False, indent=2)
    
    if excluded_results:
        with open(excluded_file, 'w', encoding='utf-8') as f:
            json.dump(excluded_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SUMMARY] Screening Summary:")
    print(f"  Total processed: {len(results_all)}")
    print(f"  Accepted: {len(included_results)}")
    print(f"  Rejected: {len(excluded_results)}")
    print(f"\n[FILES] Results saved to:")
    print(f"  All results: {results_all_file}")
    if included_results:
        print(f"  Included: {included_file}")
    if excluded_results:
        print(f"  Excluded: {excluded_file}")
    print(f"  Checkpoint: {checkpoint_file}")

    return results_all
    


# ---------------------------------------------------
#      Specific for Prof. Alborzi Congress 2025
# ---------------------------------------------------

def abstract_screen_prof_alborzi(file_path: str):
    with open(file_path, encoding='utf-8') as f:
        json_file = json.load(f)

    screen_results = []
    for abstract in json_file:
        eval_data = check_abstract_whole_concurrent(abstract['Fulltext'])
        full_eval = {**abstract, **eval_data}

        has_flags = bool(full_eval.get('flagged'))
        wrong_language = not full_eval.get('language', True)
        wrong_study_type = not full_eval.get('accurate_study_type', True)
        
        full_eval['screen_rejection'] = has_flags or wrong_language or wrong_study_type
        screen_results.append(full_eval)

    with open('screening_results.json', 'w', encoding='utf-8') as f:
        json.dump(screen_results, f, indent=2)

    return screen_results