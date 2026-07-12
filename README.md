# Sai Gon Food Chatbot

Do an nay duoc build bang Python + Streamlit de mo phong lo trinh nang cap tu local RAG len Adaptive RAG theo huong mien phi. Ban hien tai van chay tot tren may CPU, khong can GPU va co 2 che do de demo truc tiep:

- `Baseline RAG`: TF-IDF + chunk retrieval + template answer
- `Agentic RAG`: query analysis + multi-query retrieval + filter + rerank + trace tung buoc
- `Adaptive RAG`: tu route giua local retrieval, web retrieval va hybrid retrieval
- `Local LLM mode`: co the goi `Ollama` khi demo local va fallback ve template khi Ollama khong san sang
- `Conversation memory`: hieu cac cau hoi noi tiep
- `Tavily web search`: mo rong sang khu vuc ngoai dataset khi can

## Tinh nang

- Doc du lieu tu `data/*.txt`, `data/*.csv`, `data/*.json`
- Tach document, chunk va gan category tu dong
- Retrieve o muc chunk-level bang `TF-IDF + cosine similarity`
- Phan tich truy van de tim category, mon an, vibe, dia diem
- Chay vong lap agent nhe de thu nhieu bien the truy van
- Route theo kieu adaptive bang `LangChain Runnable`
- Tim kiem web bang `Tavily`
- Hien thi prompt preview va trace tung buoc cua agent
- Deploy truc tiep len Streamlit Community Cloud ma khong can API tra phi
- Ho tro `hybrid retrieval` bang cach ket hop lexical va latent semantic scoring
- Co bo `eval` de so sanh `Baseline RAG` va `Agentic RAG`

## Cau truc

```text
app.py
rag/
  agent.py
  chatbot.py
  adaptive_rag.py
  ingest.py
  pipeline.py
  query_router.py
  retriever.py
  types.py
  web_search.py
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

## Che do tim kiem

App hien co 3 che do tim kiem:

- `adaptive`: tu quyet dinh dung local, web hay ket hop
- `local_only`: chi dung du lieu RAG noi bo
- `web_only`: bo qua dataset noi bo, chi tim web

Che do `adaptive` va `web_only` can internet vi se dung Tavily de mo rong truy van theo thoi gian thuc.

## Cau hinh an toan khi deploy

- Chi set `TAVILY_API_KEY` trong Streamlit Secrets hoac bien moi truong he thong.
- Co the bat dau tu file mau `.streamlit/secrets.example.toml`.
- Khong dua file `.env` len production.
- Dat `APP_ENV=production` va `APP_DEBUG=false` khi deploy that.
- Neu muon lay debug khi dev local, co the bat `APP_DEBUG=true`.
- `ALLOW_DOTENV` mac dinh chi dung cho local dev; production se khong tu doc `.env`.

## Chay voi Ollama khi demo local

1. Cai Ollama tren may.
2. Tai model nhe, vi du:

```bash
ollama pull gemma3:1b
```

3. Chay Ollama local.
4. Mo app Streamlit va chon:
   - `Che do generation = ollama`
   - `Che do chat = agentic`
   - `Chien luoc retrieval = hybrid`

Neu Ollama khong chay hoac model khong san sang, app se tu fallback ve `template`.

## Build lai chi muc

```bash
python scripts/build_index.py
```

## Dinh dang du lieu duoc ho tro

### 1. TXT danh sach quan an

Dang tuong thich voi file `data/train.txt` hien tai.

### 2. CSV

Dat file vao `data/*.csv` voi cac cot toi thieu:

```text
title,address,description
```

Co the bo sung:

```text
doc_id,category
```

Vi du da co san:

- `data/restaurants_extra.csv`

### 3. JSON FAQ

Dat file vao `data/*.json` theo dang:

```json
[
  {
    "doc_id": "faq-1",
    "question": "Quan nao phu hop cho bua toi lang man o bo song?",
    "answer": "Ban co the uu tien ...",
    "category": "lang_man"
  }
]
```

Vi du da co san:

- `data/faq.json`

## Chay eval de lay so lieu bao cao

```bash
python scripts/run_eval.py
```

Script se xuat JSON gom 3 cau hinh:

- `baseline + lexical`
- `baseline + hybrid`
- `agentic + hybrid`

Dua cac chi so sau vao bao cao:

- `top1_accuracy`
- `topk_accuracy`
- `mrr`

## Cach demo trong bao cao

1. Mo app va de che do `Agentic RAG`.
2. Dat cau hoi nhu `Quan nao lang man cho buoi toi o TP.HCM?`
3. Mo muc `Xem flow RAG / agent trace`.
4. Giai thich 3 diem:
   - Agent phan tich truy van
   - Agent retrieve nhieu bien the thay vi 1 lan
   - Agent rerank ket qua truoc khi tra loi
5. Chuyen sang `Baseline RAG` de so sanh.
6. Neu can so lieu dinh luong, chay `python scripts/run_eval.py` va dua ket qua vao slide.
7. Neu bao cao truc tiep tren laptop, bat `generation = ollama` de the hien nang luc sinh cau tra loi tu local LLM.

## Cach demo multi-turn conversation

Thu tu goi y:

1. `Quan nao lang man cho buoi toi o TP.HCM?`
2. `Con quan nao khac?`
3. `Binh Thanh thi sao?`
4. `So sanh 2 quan nay`

Khi demo, mo `Xem flow RAG / agent trace` va nhan vao:

- `Memory debug`
- `Memory snapshot`
- `Cau hoi hieu dung de retrieve`

De giai thich rang agent da dung ngu canh hoi thoai truoc de rewrite truy van.

## Adaptive RAG route

Adaptive RAG hien route theo 3 nhom:

- `local`: khi truy van khop manh voi du lieu noi bo
- `web`: khi cau hoi nam ngoai dataset hoac nguoi dung buoc tim web
- `hybrid`: khi local co tin hieu nhung chua du manh, can mo rong them ngoai data

Dieu nay giup chatbot tra loi hop ly hon khi nguoi dung hoi cac khu vuc ngoai TP.HCM thay vi co dinh bi tu choi nhu ban cu.

Vi du nhung cau hoi se bi tu choi:

- `Quan an ngon o Ha Noi?`
- `Tim dia chi quan an o Da Nang`
- `Goi y quan o Quan 2` neu dataset hien tai chua co du lieu khu vuc do

Neu bat `search = adaptive`, cac cau hoi tren co the duoc mo rong tim kiem thay vi bi tu choi ngay.

## Deploy len Streamlit Community Cloud

1. Day source code len GitHub.
2. Dang nhap Streamlit Community Cloud.
3. Chon `New app`.
4. Tro toi repo va file `app.py`.
5. Deploy truc tiep voi `requirements.txt` hien tai.
6. Khi deploy cloud, giu `generation = template` vi Streamlit Cloud khong chay local Ollama.
7. Dat secret trong Streamlit Secrets, khong commit `.env`.

Neu sua du lieu trong `data/train.txt`, hay build lai index truoc khi commit:

```bash
python scripts/build_index.py
```

## Huong nang cap tiep theo

- Them local LLM qua Ollama cho answer generation
- Mo rong ingest cho PDF, CSV, FAQ, web data
- Them ground-truth lon hon trong `evals/` de danh gia nghiem tuc hon
- Them cache va timeout thong minh cho Ollama
- Them bo nho dai han co tom tat hoi thoai tot hon
- Them parser PDF khi can mo rong len tai lieu dai
