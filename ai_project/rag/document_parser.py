# document_parser.py
import pdfplumber
import re
from config.config import CHUNK_MAX_LENGTH, CHUNK_OVERLAP

def semantic_overlap_chunking(text, max_length=CHUNK_MAX_LENGTH, overlap_sentences=CHUNK_OVERLAP):
    clean_text = text.replace('\n', '')
    clean_text = re.sub(r'\s+', ' ', clean_text)
    sentences = re.split(r'(?<=[。！？])', clean_text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if not sentence.strip(): continue
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            if current_chunk: chunks.append("".join(current_chunk))
            overlap_part = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap_part + [sentence]
            current_length = sum(len(s) for s in current_chunk)
            
    if current_chunk: chunks.append("".join(current_chunk))
    return chunks

def load_and_chunk_pdf(pdf_path):
    print(f"📖 正在解析 {pdf_path} ...")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text: full_text += text + " "
    
    chunks = semantic_overlap_chunking(full_text)
    print(f"✅ 成功切分出 {len(chunks)} 块文档。")
    return chunks