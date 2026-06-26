# Sai Gon Food Chatbot

Chatbot nay duoc build bang Python + Streamlit de mo phong giao dien chat kieu Messenger va giup ban hoc flow RAG co ban. Phien ban hien tai dung `TF-IDF + cosine similarity` de retrieval o muc `chunk-level` va `template answer` de sinh cau tra loi, nen chay tot tren may CPU khong can GPU.

## Tinh nang

- Doc du lieu tu `data/*.txt`
- Tach moi muc thanh document co tieu de, dia chi, noi dung, nhom chu de
- Cat document thanh nhieu chunk nho co overlap nhe de retrieval sat RAG hon
- Tao chi muc retrieval cuc bo bang TF-IDF tren tung chunk
- Hien thi cau tra loi, nguon tham khao va debug flow RAG
- Giao dien Streamlit chat don gian, de doc va de sua

## Cau truc

```text
app.py
rag/
  ingest.py
  retriever.py
  chatbot.py
  pipeline.py
  types.py
scripts/
  build_index.py
data/
```

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chay app

```bash
streamlit run app.py
```

## Build lai chi muc

```bash
python scripts/build_index.py
```

## Cach hoc flow

1. Dat cau hoi trong giao dien chat.
2. Mo muc `Xem flow RAG`.
3. Quan sat cac chunk duoc retrieve, diem lien quan va prompt preview.
4. Sua du lieu trong `data/train.txt` va build lai de thay doi hanh vi chatbot.
