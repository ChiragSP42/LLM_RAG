from tqdm.auto import tqdm
import wikipedia
from dotenv import load_dotenv
import os
import huggingface_hub
import chromadb
from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

financial_stock_terms = [
 'Stock',
 'Share',
 'Market Cap',
 'Dividend',
 'P/E Ratio',
 'EPS',
 'Bull Market',
 'Bear Market',
 'Volume',
 'Volatility',
 'Blue Chip',
 'NASDAQ',
 'S&P 500',
 'Dow Jones',
 'Portfolio',
 'Diversification',
 'Asset Allocation',
 'Broker',
 'Commission',
 'Limit Order',
 'Market Order',
 'Day Trading',
 'Short Selling',
 'Margin',
 'Leverage',
 'Bid-Ask Spread',
 'Beta',
 'Alpha',
 'Fundamental Analysis',
 'Moving Average',
 'Resistance Level',
 'Support Level',
 'Trend Line',
 'Breakout',
 'Pullback',
 'Rally',
 'Correction',
 'Sector',
 'Industry',
 'ETF',
 'Mutual Fund',
 'Index Fund',
 'REIT',
 'ADR',
 'Futures',
 'Options',
 'Call Option',
 'Put Option',
 'Strike Price',
 'Expiration Date',
 'Derivative',
 'Hedge',
 'Arbitrage',
 'Earnings Report',
 'Balance Sheet',
 'Income Statement',
 'Cash Flow Statement',
 'Quarterly Report',
 'Annual Report',
 '10-K',
 '10-Q',
 'Price Target',
 'Upgrade',
 'Downgrade',
 'Overweight',
 'Market Maker',
 'Dark Pool',
 'High-Frequency Trading',
 'Algorithmic Trading',
 'Front-Running',
 'Penny Stock',
 'Growth Stock',
 'Value Stock',
 'Cyclical Stock',
 'Preferred Stock',
 'Common Stock',
 'Warrant',
 'Rights Issue',
 'Stock Split',
 'Reverse Split',
 'Buyback',
 'Dilution',
 'Outstanding Shares',
 'Treasury Stock',
 'Ex-Dividend Date',
 'Dividend Reinvestment Plan (DRIP)',
 'Tax-Loss Harvesting',
 'Wash Sale']

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "stock_terms"
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=True)

chroma_embedding_model = SentenceTransformer("sentence-transformers/sentence-t5-base")
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

contexts = {}

for term in tqdm(financial_stock_terms):
    try:
        search_results = wikipedia.search(term)
        page = wikipedia.page(search_results)
        contexts[term] = page.summary
    except:
        print(f"Page {term} not found")


collection.add(
    documents=list(contexts.values()),
    ids=[f"id{i}" for i in range(len(list(contexts.values())))],
    metadatas=[{"genre": g} for g in list(contexts.keys())]
)

query_results = collection.query(
    query_texts=["What are stocks?"],
    n_results=1,
)

print(query_results["documents"][0])
