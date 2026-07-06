# Sai Gon Food Chatbot

Do an nay duoc build bang Python + Streamlit de mo phong lo trinh nang cap tu local RAG len agentic RAG theo huong mien phi. Ban hien tai van chay tot tren may CPU, khong can GPU va co 2 che do de demo truc tiep:

- `Baseline RAG`: TF-IDF + chunk retrieval + template answer
- `Agentic RAG`: query analysis + multi-query retrieval + filter + rerank + trace tung buoc

## Tinh nang

- Doc du lieu tu `data/*.txt`
- Tach document, chunk va gan category tu dong
- Retrieve o muc chunk-level bang `TF-IDF + cosine similarity`
- Phan tich truy van de tim category, mon an, vibe, dia diem
- Chay vong lap agent nhe de thu nhieu bien the truy van
- Hien thi nguon tham khao, prompt preview va trace tung buoc cua agent
- Deploy truc tiep len Streamlit Community Cloud ma khong can API tra phi

## Cau truc

```text
app.py
rag/
  agent.py
  chatbot.py
  ingest.py
  pipeline.py
  query_router.py
  retriever.py
  types.py
scripts/
  build_index.py
data/
artifacts/
```

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chay app local

```bash
streamlit run app.py
```

## Build lai chi muc

```bash
python scripts/build_index.py
```

## Cach demo trong bao cao

1. Mo app va de che do `Agentic RAG`.
2. Dat cau hoi nhu `Quan nao lang man cho buoi toi o TP.HCM?`
3. Mo muc `Xem flow RAG / agent trace`.
4. Giai thich 3 diem:
   - Agent phan tich truy van
   - Agent retrieve nhieu bien the thay vi 1 lan
   - Agent rerank ket qua truoc khi tra loi
5. Chuyen sang `Baseline RAG` de so sanh.

## Deploy len Streamlit Community Cloud

1. Day source code len GitHub.
2. Dang nhap Streamlit Community Cloud.
3. Chon `New app`.
4. Tro toi repo va file `app.py`.
5. Deploy truc tiep voi `requirements.txt` hien tai.

Neu sua du lieu trong `data/train.txt`, hay build lai index truoc khi commit:

```bash
python scripts/build_index.py
```

## Huong nang cap tiep theo

- Them dense embedding de thanh hybrid RAG thuc su
- Them local LLM qua Ollama cho answer generation
- Them bo eval va benchmark giua baseline va agentic
- Mo rong ingest cho PDF, CSV, FAQ, web data
