import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).parent

# Only force offline HuggingFace Hub mode if a bundled local `./models/hub` cache exists.
# Otherwise, allow HuggingFace Hub to download and cache models normally (~/.cache/huggingface).
_LOCAL_HUB_DIR = BASE_DIR / "models" / "hub"
if _LOCAL_HUB_DIR.exists():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(_LOCAL_HUB_DIR)

DATA_DIR = BASE_DIR / "data_pro_tier"
DIR_400M = BASE_DIR / "data_fast_tier"

FAST_INDEX = DIR_400M / "data" / "mrl_index" / "index.faiss"
FAST_IDMAP = DIR_400M / "data" / "mrl_index" / "id_map.json"

PRO_INDEX  = DATA_DIR / "mrl_index" / "index.faiss"
PRO_IDMAP  = DATA_DIR / "mrl_index" / "id_map.json"
PRO_CACHE  = DATA_DIR / "mrl_cache.pkl"
HYDE_CACHE = DATA_DIR / "hyde_cache.json"

SCIFACT_DIR  = BASE_DIR / "scifact_data"
CORPUS_FILE  = SCIFACT_DIR / "corpus.jsonl"
QUERIES_FILE = SCIFACT_DIR / "queries.jsonl"
QRELS_FILE   = SCIFACT_DIR / "qrels" / "test.tsv"

MRL_DIM     = 256
STAGE1_K    = 50
TOP_K       = 10
APP_VERSION = "V3.29"

GITHUB_RELEASE_CACHE_URL = (
    "https://github.com/Ceviche98/scifact-tiered-neural-search/releases/download/v1.0.0/mrl_cache.pkl"
)

EXAMPLE_QUERIES = [
    "Bariatric surgery has a positive impact on mental health.",
    "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.",
    "A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.",
    "Cell autonomous sex determination in somatic cells occurs in Passeriformes.",
    "Chronic aerobic exercise alters endothelial function, improving vasodilating mechanisms mediated by NO.",
    "A deficiency of vitamin B12 increases blood levels of homocysteine.",
    "ADAR1 binds to Dicer to cleave pre-miRNA.",
    "Cataract and trachoma are the primary cause of blindness in Southern Sudan.",
    "Cellular aging closely links to an older appearance.",
    "Chenodeosycholic acid treatment increases whole-body energy expenditure.",
]
