import streamlit as st
import pypdf
import re
import os
import hashlib
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from flashrank import Ranker, RerankRequest
import google.generativeai as genai

# =====================================================================
# 1. DATA CONTRACT INTERFACES
# =====================================================================
@dataclass
class RankedChunk:
    id: str
    content: str
    score: float
    metadata: dict

@dataclass
class LLMResponse:
    answer: str
    reasoning: str
    input_tokens: int
    output_tokens: int

@dataclass
class PipelineResult:
    status: str
    answer: str
    warning: str
    top_chunks: list[RankedChunk]
    latency_ms: dict

# =====================================================================
# 2. LOCAL DOCUMENT STORAGE & CHROMADB MANAGEMENT
# =====================================================================
class DocumentManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except Exception:
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            self.client = chromadb.PersistentClient(path=persist_directory)
            
        self.collection = self.client.get_or_create_collection(
            name="regulatory_manuals",
            metadata={"hnsw:space": "cosine"}
        )

    def _make_chunk_id(self, filename: str, page_number: int, chunk_idx: int) -> str:
        raw_key = f"{filename}::page::{page_number}::chunk::{chunk_idx}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def ingest_pdf(self, file_path: str, filename: str):
        reader = pypdf.PdfReader(file_path)
        chunks_to_add = []
        metadatas = []
        ids = []
        
        chunk_idx = 0
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text()
            if not raw_text:
                continue
                
            clean_text = re.sub(r'\s+', ' ', raw_text).strip()
            
            start = 0
            while start < len(clean_text):
                end = start + 1000
                chunk_content = clean_text[start:end]
                
                c_id = self._make_chunk_id(filename, page_num, chunk_idx)
                chunks_to_add.append(chunk_content)
                metadatas.append({
                    "source_file": filename,
                    "page_number": page_num
                })
                ids.append(c_id)
                
                start += 800  
                chunk_idx += 1
                
        if chunks_to_add:
            embeddings = self.embedder.encode(chunks_to_add, convert_to_numpy=True).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=chunks_to_add,
                metadatas=metadatas,
                ids=ids
            )
        return len(chunks_to_add)

    def broad_vector_search(self, query: str, n_results=20) -> list[dict]:
        if self.collection.count() == 0:
            return []
        query_vector = self.embedder.encode([query], convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=min(n_results, self.collection.count())
        )
        
        parsed_results = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                parsed_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })
        return parsed_results

    def get_ingested_files(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        try:
            results = self.collection.get(include=["metadatas"])
            if not results or not results.get('metadatas'):
                return []
            files = set()
            for meta in results['metadatas']:
                if meta and "source_file" in meta:
                    files.add(meta["source_file"])
            return sorted(list(files))
        except Exception:
            return []

# =====================================================================
# 3. STAGE-TWO CROSS-ENCODER RERANKER
# =====================================================================
class StageTwoReranker:
    def __init__(self):
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def execution_rerank(self, query: str, candidates: list[dict], top_n=5) -> list[RankedChunk]:
        if not candidates:
            return []
            
        passages = [
            {"id": c["id"], "text": c["content"], "meta": c["metadata"]}
            for c in candidates
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        final_ranked = []
        for r in results[:top_n]:
            final_ranked.append(RankedChunk(
                id=r["id"],
                content=r["text"],
                score=float(r["score"]),
                metadata=r["meta"]
            ))
        return final_ranked

# =====================================================================
# 4. LLM INFERENCE LAYER & BOUNDED GUARDRAILS
# =====================================================================
GEMINI_SECURE_KEY = "AQ.Ab8RN6KVMCIMNAL9VkDVP55p5ycyCOB1_Yhdrqoxrm0cNAUo5Q"

class LLMService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def synthesize(self, query: str, context_chunks: list[RankedChunk]) -> LLMResponse:
        accumulated_context = ""
        for chunk in context_chunks:
            if len(accumulated_context) + len(chunk.content) < 6000:
                accumulated_context += f"\n---\nSource: {chunk.metadata['source_file']} (Page {chunk.metadata['page_number']})\nText: {chunk.content}\n"
            else:
                break

        system_prompt = (
            "You are a deterministic compliance auditing agent. Answer the user's query BASED STRICTLY "
            "on the verified context blocks provided below. Do not use external knowledge or speculate. "
            "If the answer cannot be explicitly verified from the context, respond with: 'The provided context does not contain this information.'\n"
            "Separate your internal logical verification steps inside markdown tags like <reasoning>...</reasoning>.\n"
        )
        
        prompt_payload = f"{system_prompt}\n[VERIFIED CONTEXT]\n{accumulated_context}\n\n[QUERY]\n{query}"
        
        response = self.model.generate_content(prompt_payload)
        text = response.text
        
        reasoning = ""
        answer = text
        if "<reasoning>" in text and "</reasoning>" in text:
            reasoning = text.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            answer = text.split("</reasoning>")[1].strip()

        return LLMResponse(
            answer=answer,
            reasoning=reasoning,
            input_tokens=getattr(response.usage_metadata, 'prompt_token_count', 0),
            output_tokens=getattr(response.usage_metadata, 'candidates_token_count', 0)
        )

# =====================================================================
# 5. ORCHESTRATION PIPELINE LOGIC WITH FAILURE ROUTING
# =====================================================================
class RAGPipeline:
    def __init__(self, doc_manager: DocumentManager, reranker: StageTwoReranker):
        self.doc_manager = doc_manager
        self.reranker = reranker

    def run(self, query: str, force_fail_toggle: bool) -> PipelineResult:
        import time
        latencies = {}
        
        t0 = time.time()
        candidates = self.doc_manager.broad_vector_search(query, n_results=20)
        latencies["retrieval_ms"] = (time.time() - t0) * 1000
        
        t1 = time.time()
        top_chunks = self.reranker.execution_rerank(query, candidates, top_n=5)
        latencies["rerank_ms"] = (time.time() - t1) * 1000
        
        latencies["llm_ms"] = 0.0
        
        if force_fail_toggle:
            return PipelineResult(
                status="LLM_FALLBACK",
                answer="",
                warning="CRITICAL FALLBACK ACTIVATED: API Failure Simulation Active. Engaging Partial RAG Mode Failsafe.",
                top_chunks=top_chunks,
                latency_ms=latencies
            )

        try:
            llm_service = LLMService(GEMINI_SECURE_KEY)
            t2 = time.time()
            llm_resp = llm_service.synthesize(query, top_chunks)
            latencies["llm_ms"] = (time.time() - t2) * 1000
            
            return PipelineResult(
                status="SUCCESS",
                answer=llm_resp.answer,
                warning=f"<reasoning>{llm_resp.reasoning}</reasoning>" if llm_resp.reasoning else "",
                top_chunks=top_chunks,
                latency_ms=latencies
            )
        except Exception as e:
            return PipelineResult(
                status="LLM_FALLBACK",
                answer="",
                warning=f"EXCEPTION INTERCEPTED: Generative Remote API Timeout ({str(e)}). Graceful Fallback Render Engaged.",
                top_chunks=top_chunks,
                latency_ms=latencies
            )

# =====================================================================
# 6. STREAMLIT PREMIUM VISUAL LAYOUT (HIGH FIDELITY THEME OVERRIDE)
# =====================================================================
st.set_page_config(page_title="RAG Pipeline", layout="wide")

# Injection of state-of-the-art Dribbble-inspired light-dashboard styles
# Injection of state-of-the-art Dribbble-inspired dark-dashboard styles (without blank lines)
st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"><style>html, body, [data-testid="stAppViewContainer"] {background-color: #050811 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; color: #f1f5f9 !important;} label, div[data-testid="stWidgetLabel"] p, .stMarkdown p, .stMarkdown h3, .stMarkdown h2, .stMarkdown h1, [data-testid="stFileUploader"] p {color: #ffffff !important;} [data-testid="stHeader"] {background: transparent !important; background-color: transparent !important;} [data-testid="stSidebar"] {background-color: #03050a !important; border-right: 1px solid #111827 !important; padding-top: 15px !important;} .header-banner {background: linear-gradient(135deg, #090d16 0%, #0d1527 100%); border: 1px solid #111827; border-left: 6px solid #10b981; padding: 24px 30px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);} [data-testid="stTabs"] {background-color: #090d16 !important; border-radius: 12px !important; padding: 24px !important; border: 1px solid #111827 !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2) !important; margin-bottom: 25px !important;} button[data-testid="stTab"] {font-size: 14px !important; font-weight: 600 !important; color: #94a3b8 !important; padding: 10px 18px !important; border-radius: 8px !important; transition: all 0.2s ease !important; border: none !important;} button[data-testid="stTab"][aria-selected="true"] {background-color: #111827 !important; color: #10b981 !important;} button[data-testid="stTab"]:hover {color: #10b981 !important;} .stTextInput>div>div>input {background-color: #0d121f !important; border: 1px solid #1e293b !important; color: #ffffff !important; border-radius: 8px !important; padding: 10px 14px !important; font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 500 !important; transition: all 0.2s ease !important;} .stNumberInput [data-testid="stNumberInputContainer"], .stNumberInput [data-testid="stNumberInputContainer"] div, .stNumberInput [data-testid="stNumberInputContainer"] input, .stNumberInput [data-testid="stNumberInputContainer"] button {background-color: #0d121f !important; color: #ffffff !important; border: none !important;} .stNumberInput [data-testid="stNumberInputContainer"] {border: 1px solid #1e293b !important; border-radius: 8px !important;} .stNumberInput button:hover {background-color: rgba(255,255,255,0.05) !important; color: #10b981 !important;} .stTextInput input, .stNumberInput input, .stTextInput input:focus, .stNumberInput input:focus, .stNumberInput input:active {color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;} .stTextInput input::placeholder, .stNumberInput input::placeholder {color: rgba(255,255,255,0.75) !important; -webkit-text-fill-color: rgba(255,255,255,0.75) !important; opacity: 1 !important;} .stTextInput>div>div>input:focus {border-color: #10b981 !important; background-color: #090d16 !important; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;} [data-testid="stFileUploader"] {border: 2px dashed #1e293b !important; border-radius: 10px !important; background-color: #0d121f !important; padding: 16px !important; transition: all 0.2s ease !important;} [data-testid="stFileUploader"]:hover {border-color: #10b981 !important;} [data-testid="stFileUploader"] button, [data-testid="stFileUploader"] [role="button"] {background: linear-gradient(135deg, #1e40af, #2563eb) !important; color: #ffffff !important; border: 1px solid #3b82f6 !important; border-radius: 8px !important; font-weight: 600 !important; transition: all 0.2s ease !important; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25) !important;} [data-testid="stFileUploader"] button:hover, [data-testid="stFileUploader"] [role="button"]:hover {background: #1d4ed8 !important; border-color: #2563eb !important; transform: translateY(-1px) !important;} [data-testid="stFileUploader"] button *, [data-testid="stFileUploader"] [role="button"] * {color: #ffffff !important; fill: #ffffff !important;} .stButton>button {background: linear-gradient(135deg, #22c55e, #10b981) !important; color: #0b0f19 !important; font-weight: 800 !important; font-size: 14px !important; border-radius: 8px !important; border: none !important; padding: 12px 24px !important; box-shadow: 0 4px 14px 0 rgba(34, 197, 94, 0.3) !important; transition: all 0.2s ease-in-out !important; width: auto !important; display: inline-flex !important;} .stButton>button:hover {transform: translateY(-1.5px) !important; box-shadow: 0 6px 20px 0 rgba(34, 197, 94, 0.45) !important;} .stButton>button:active {transform: translateY(0) !important;} .source-card {background-color: #090d16; border: 1px solid #111827; padding: 20px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2); transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);} .source-card:hover {transform: translateY(-2px); box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.3); border-color: #1e293b !important;} .metric-container-box {background-color: #090d16; border: 1px solid #111827; border-radius: 10px; padding: 14px; margin-bottom: 12px; text-align: center; box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1); transition: all 0.2s ease;} .metric-container-box:hover {transform: scale(1.02); box-shadow: 0 4px 8px 0 rgba(0,0,0,0.15);} .profile-card-box {background-color: #090d16; border: 1px solid #111827; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 24px; box-shadow: 0 4px 10px 0 rgba(0,0,0,0.15);} .profile-card-box a div:hover {background-color: #005582 !important; transform: translateY(-1.5px); box-shadow: 0 6px 14px rgba(0, 119, 181, 0.35) !important;} [data-testid="stAlert"] {border-radius: 10px !important; border: 1px solid #111827 !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;} [data-testid="stExpander"], [data-testid="stExpander"] details, [data-testid="stExpander"] summary, [data-testid="stExpander"] summary > div, .streamlit-expanderHeader, .streamlit-expanderHeader > div {background-color: #090d16 !important; border: 1px solid #111827 !important; border-radius: 8px !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2) !important; margin-bottom: 15px !important;} [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary svg, [data-testid="stExpander"] p, [data-testid="stExpander"] span, .streamlit-expanderHeader, .streamlit-expanderHeader p, .streamlit-expanderHeader svg, [data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] span {color: #ffffff !important; fill: #ffffff !important;} [data-testid="stSidebarCollapseButton"] button, [data-testid="collapsedControl"] button, [data-testid="stSidebarCollapseButton"] button svg, [data-testid="collapsedControl"] button svg {color: #ffffff !important; fill: #ffffff !important; stroke: #ffffff !important;}</style>""", unsafe_allow_html=True)

if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if "reranker" not in st.session_state:
    st.session_state.reranker = StageTwoReranker()

pipeline = RAGPipeline(st.session_state.doc_manager, st.session_state.reranker)

# 📂 SIDEBAR CONTROL PANEL
with st.sidebar:
    # Centered Chanakya University Logo Display Banner
    col_logo_1, col_logo_2, col_logo_3 = st.columns([1, 8, 1])
    with col_logo_2:
        if os.path.exists("chanakya_logo.png"):
            st.image("chanakya_logo.png", use_container_width=True)
        else:
            st.markdown("""<div style="text-align:center; margin-bottom:15px; background:white; padding:10px; border-radius:8px; border: 1px solid #e2e8f0;"><h3 style="color:#1e3a8a; margin:0; font-family:sans-serif; font-weight:bold; letter-spacing: 1px;">CHANAKYA</h3><p style="color:#64748b; margin:0; font-size:11px; letter-spacing:2px;">UNIVERSITY</p></div>""", unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)

    # Developer Credentials Badge Card
    st.markdown("""<div class="profile-card-box"><div style="width: 48px; height: 48px; background: linear-gradient(135deg, #4f46e5, #6366f1); color: #ffffff; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; margin: 0 auto 12px auto; box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);">RS</div><div style="font-size: 10px; color: #10b981; font-weight: 700; letter-spacing: 1.5px;">PROJECT DEVELOPER</div><div style="font-size: 18px; font-weight: 700; color: #f8fafc; margin-top: 4px;">Rishav Singh</div><div style="font-size: 13.5px; color: #94a3b8; margin-top: 2px;">M.Sc. Data Science</div><div style="font-size: 11px; color: #64748b;">Batch 2025–2027</div><div style="margin-top: 15px;"><a href="https://www.linkedin.com/in/rishav-singh-a61964248" target="_blank" style="text-decoration: none;"><div style="background-color: #0077b5; color: #ffffff; border-radius: 20px; padding: 8px 16px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; gap: 8px; box-shadow: 0 4px 8px rgba(0, 119, 181, 0.2); transition: all 0.2s ease;"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" viewBox="0 0 24 24" style="margin-bottom: 1px;"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.779-1.75-1.75s.784-1.75 1.75-1.75 1.75.779 1.75 1.75-.784 1.75-1.75 1.75zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>Connect on LinkedIn</div></a></div></div>""", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Controls")
    fail_simulation = st.toggle("Simulate API Failure")
    
    if fail_simulation:
        st.markdown('<div style="background-color: #2d1515; border: 1px solid #7f1d1d; padding: 12px; border-radius: 8px; font-size: 12px; color: #fca5a5; font-weight: 500; display: flex; align-items: center; gap: 8px;"><span style="color:#ef4444; font-size:16px;">⚠️</span> <b>Fallback Enforced</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color: #062f22; border: 1px solid #064e3b; padding: 12px; border-radius: 8px; font-size: 12px; color: #86efac; font-weight: 500; display: flex; align-items: center; gap: 8px;"><span style="color:#22c55e; font-size:16px;">🟢</span> <b>API Connected</b></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 📊 Last Run Metrics")
    
    ret_stat = st.empty()
    rnk_stat = st.empty()
    llm_stat = st.empty()
    
    ret_stat.markdown('<div class="metric-container-box" style="border-left: 4px solid #1e293b;"><div style="color:#64748b; font-size:11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Stage 1 Recall</div><div style="font-size:16px; font-weight: 700; color:#475569; margin-top: 4px;">0.00 ms</div></div>', unsafe_allow_html=True)
    rnk_stat.markdown('<div class="metric-container-box" style="border-left: 4px solid #1e293b;"><div style="color:#64748b; font-size:11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Stage 2 Rerank</div><div style="font-size:16px; font-weight: 700; color:#475569; margin-top: 4px;">0.00 ms</div></div>', unsafe_allow_html=True)
    llm_stat.markdown('<div class="metric-container-box" style="border-left: 4px solid #1e293b;"><div style="color:#64748b; font-size:11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Stage 3 Generation</div><div style="font-size:16px; font-weight: 700; color:#475569; margin-top: 4px;">0.00 ms</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    # Ingested Files Section
    st.markdown("### 📁 Ingested Files")
    ingested_files = st.session_state.doc_manager.get_ingested_files()
    with st.expander("Show all ingested files", expanded=False):
        if ingested_files:
            for f in ingested_files:
                st.markdown(f"📄 **{f}**")
        else:
            st.markdown("*No files ingested yet*")

# 💻 MAIN INTERFACE AREA
st.markdown("""<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 style="margin:0; font-size:26px; font-weight:800; color:#ffffff; letter-spacing: -0.5px; font-family:sans-serif;">Enterprise RAG Auditor</h1>
            <p style="margin:6px 0 0 0; color:#94a3b8; font-size:12px; font-weight: 500;">Chanakya University &bull; Document Manager &bull; FlashRank &bull; Gemini 2.5 Flash</p>
        </div>
        <div style="background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.2); padding: 6px 14px; border-radius: 20px; color: #10b981; font-size: 11px; font-weight: 600;">Production Ready RAG v2.0</div>
    </div>
</div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📥 Ingest Documents", "💬 Query"])

with tab1:
    st.markdown("### Upload & Index Documents")
    uploaded_file = st.file_uploader("Choose one or more .txt, .pdf, .docx files", type=["pdf"], label_visibility="visible")
    
    col1, col2 = st.columns(2)
    with col1:
        doc_id_opt = st.text_input("Document ID (optional)", placeholder="e.g. 101")
    with col2:
        st.number_input("Words per page", value=500, step=50)
        
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    if st.button("📁 Ingest Files"):
        if uploaded_file:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Processing local vector integration layout..."):
                total_added = st.session_state.doc_manager.ingest_pdf(temp_path, uploaded_file.name)
                st.success(f"Successfully tracked '{uploaded_file.name}' inside ChromaDB! Index space contains {total_added} standalone blocks.")
            os.remove(temp_path)
        else:
            st.warning("Please upload a PDF document manual first.")

    # =====================================================================
    # CHROMADB CRUD INTERACTIVE MANAGEMENT UNIT
    # =====================================================================
    st.markdown("---")
    st.markdown("### 🛠️ Database Index Management (CRUD)")
    
    # 1. READ ACTIVE DOCUMENTS
    all_data = st.session_state.doc_manager.collection.get()
    existing_docs = sorted(list(set([m["source_file"] for m in all_data["metadatas"]]))) if all_data and all_data["metadatas"] else []
    
    if existing_docs:
        st.markdown(f"**Current Active Collections Count:** `{len(existing_docs)} unique files indexed`")
        
        # 2. UPDATE METADATA
        with st.expander("📝 Update Indexed Filename Metadata"):
            target_update = st.selectbox("Select Target File to Rename", existing_docs, key="crud_update_select")
            new_name = st.text_input("Enter New Filename Metadata Entry:", value=target_update)
            
            if st.button("Apply Metadata Change"):
                ids_to_update = [all_data["ids"][i] for i, m in enumerate(all_data["metadatas"]) if m["source_file"] == target_update]
                metas_to_update = [all_data["metadatas"][i] for i, m in enumerate(all_data["metadatas"]) if m["source_file"] == target_update]
                for meta in metas_to_update:
                    meta["source_file"] = new_name
                st.session_state.doc_manager.collection.update(ids=ids_to_update, metadatas=metas_to_update)
                st.success(f"Updated metadata mappings from '{target_update}' to '{new_name}'!")
                st.rerun()

        # 3. DELETE DOCUMENTS
        with st.expander("🗑️ Delete Selected Document Entries"):
            target_delete = st.selectbox("Select Target File to Purge", existing_docs, key="crud_delete_select")
            if st.button("Wipe File From Index"):
                ids_to_delete = [all_data["ids"][i] for i, m in enumerate(all_data["metadatas"]) if m["source_file"] == target_delete]
                st.session_state.doc_manager.collection.delete(ids=ids_to_delete)
                st.success(f"Successfully removed '{target_delete}' vectors from local instance.")
                st.rerun()
    else:
        st.info("ChromaDB vector instance storage is currently empty. Ingest a document manual to open CRUD tools.")

with tab2:
    st.markdown("### Run Bounded Compliance Audits")
    query_str = st.text_input("Enter your validation tracking query statement here:", placeholder="e.g. Type audit question criteria...")
    
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    if st.button("🚀 Run Assessment Pass"):
        if not query_str:
            st.warning("Query text input string empty.")
        else:
            with st.spinner("Processing Context Selection Optimization Layer..."):
                res = pipeline.run(query_str, fail_simulation)
                
                # Dynamic performance grid latency metrics injection update
                ret_stat.markdown(f"""<div class="metric-container-box" style="border-left: 4px solid #38bdf8;">
                    <div style="color: #94a3b8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Stage 1 Recall</div>
                    <div style="font-size: 16px; font-weight: 700; color: #38bdf8; margin-top: 4px;">{res.latency_ms["retrieval_ms"]:.2f} ms</div>
                </div>""", unsafe_allow_html=True)
                
                rnk_stat.markdown(f"""<div class="metric-container-box" style="border-left: 4px solid #a3e635;">
                    <div style="color: #94a3b8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Stage 2 Rerank</div>
                    <div style="font-size: 16px; font-weight: 700; color: #a3e635; margin-top: 4px;">{res.latency_ms["rerank_ms"]:.2f} ms</div>
                </div>""", unsafe_allow_html=True)
                
                llm_stat.markdown(f"""<div class="metric-container-box" style="border-left: 4px solid #fbbf24;">
                    <div style="color: #94a3b8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Stage 3 Generation</div>
                    <div style="font-size: 16px; font-weight: 700; color: #fbbf24; margin-top: 4px;">{res.latency_ms["llm_ms"]:.2f} ms</div>
                </div>""", unsafe_allow_html=True)
                
                if res.status == "SUCCESS":
                    if res.warning and "<reasoning>" in res.warning:
                        with st.expander("👁️ View Internal Trace Chain-of-Thought (CoT) Verification Steps"):
                            st.write(res.warning.replace("<reasoning>", "").replace("</reasoning>", ""))
                    
                    st.markdown("### 🤖 Analytical Audit Synthesis Report")
                    st.success(res.answer)
                    
                    st.markdown("### 📍 Document & Page Citing Links")
                    seen_citations = set()
                    citations_html = ""
                    for chunk in res.top_chunks:
                        citation_key = f"{chunk.metadata['source_file']}::page::{chunk.metadata['page_number']}"
                        if citation_key not in seen_citations:
                            seen_citations.add(citation_key)
                            citations_html += f"""<div style="background-color: #1e293b; border: 1px solid #334155; color: #60a5fa; padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 500; margin-bottom: 10px; display: inline-flex; align-items: center; gap: 8px; margin-right: 10px;"><span style="font-size:16px;">📄</span><span>Ref: <b style="color:#f1f5f9;">{chunk.metadata['source_file']}</b> (Page <b style="color:#f1f5f9;">{chunk.metadata['page_number']}</b>)</span></div>"""
                    if citations_html:
                        st.markdown(f'<div style="margin-bottom:20px; display: flex; flex-wrap: wrap;">{citations_html}</div>', unsafe_allow_html=True)

                elif res.status == "LLM_FALLBACK":
                    st.warning(res.warning)
                    st.markdown("### ⚠️ Displaying Partial RAG Mode Local Document Matches")
                
                st.markdown("---")
                st.markdown("### 📄 Verified Base Source Chunks (FlashRank Sorted)")
                for idx, chunk in enumerate(res.top_chunks):
                    st.markdown(f"""<div class="source-card" style="border-left: 4px solid #4f46e5;"><h5 style="margin:0 0 6px 0; color:#818cf8; font-size:14px; font-weight:700;">🗂️ Chunk Target #{idx+1} | Confidence Score: <span style="color:#34d399;">{chunk.score:.4f}</span></h5><div style="margin:0 0 12px 0; font-size:11px; color:#94a3b8; font-weight:500;"><span style="background-color:#1e293b; padding:2px 8px; border-radius:12px; margin-right:8px; color:#cbd5e1;">📄 {chunk.metadata['source_file']}</span><span style="background-color:#1e293b; padding:2px 8px; border-radius:12px; color:#cbd5e1;">Page {chunk.metadata['page_number']}</span></div><p style="font-size:13px; margin:0; color:#cbd5e1; line-height:1.6; font-family: 'Plus Jakarta Sans', sans-serif;">{chunk.content}</p></div>""", unsafe_allow_html=True)