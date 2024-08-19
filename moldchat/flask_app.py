from flask import Flask, request, jsonify
from tool.process_file import DocumentParser, TextCleaner, FileManager
from pathlib import Path
import os
from llm.llm_adapter import RAG_Chroma
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['PROPAGATE_EXCEPTIONS'] = False
project_path=Path("/data/usr/jy/chat_web/back/moldchat")
uploaded_file_path=Path("/data/usr/jy/chat_web/back/sbt/yeya/uploads")

@app.route('/process/file/extract', methods=['POST'])
def process_file():
    data = request.get_json()
    file_path = data.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400

    document_parser = DocumentParser(file_path)
    document_parser_out_path = document_parser.extract_text()["file"]
    text_cleaner = TextCleaner()
    file_name=Path(document_parser_out_path).name
    text_cleaner_out_path = str(Path(document_parser_out_path).parent / "RAG_file" / file_name)
    text_cleaner.clean(document_parser_out_path, text_cleaner_out_path)
    result = f"Processing file"
    return jsonify({"message": result, "file": document_parser_out_path})

@app.route('/rag/create', methods=['POST'])
def create_rag():
    data = request.get_json()
    db_name = data.get('name')
    db_path = str(project_path / Path("chroma_db") / db_name)
    raw_file_path =str(project_path / Path("_raw_file") / db_name)
    file_names_dict=data.get("files")
    file_names=[f["fileName"] for f in file_names_dict]
    file_manager=FileManager()
    if not os.path.exists(raw_file_path):
        os.makedirs(raw_file_path)
    file_manager.move_files(file_names, str(uploaded_file_path), raw_file_path, [".txt"])
    rag_chroma = RAG_Chroma(db_path)
    rag_chroma.create(raw_file_path, is_file_path_dir=True)
    result = f"RAG Chroma created"
    return jsonify({"msg": result})

@app.route('/rag/drop/<db_name>', methods=['DELETE'])
def drop_rag(db_name):
    db_path = str(project_path / Path("chroma_db") / db_name)
    rag_chroma = RAG_Chroma(db_path)
    res = rag_chroma.drop()
    return jsonify({"ok": res})

@app.route('/rag/query', methods=['POST'])
def query_rag():
    data = request.get_json()
    db_path = str(project_path / Path("chroma_db") / data.get('db_name'))
    rag_chroma = RAG_Chroma(db_path)
    query = data.get('query')
    top_k = data.get('top_k')
    docs = rag_chroma.query(query, top_k)
    return jsonify({"docs": docs})

@app.route('/rag/insert', methods=['POST'])
def insert_rag():
    data = request.get_json()
    db_name = data.get('db_name')
    db_path = str(project_path / Path("chroma_db") / db_name)
    rag_chroma = RAG_Chroma(db_path)
    rag_chroma.insert(str(uploaded_file_path / db_name))
    
@app.route('/rag/select', methods=['GET'])
def get_texts_rag():
    data = request.get_json()
    db_name = data.get('db_name')
    keyword = data.get("keyword", None)
    limit = data.get("limit", None)
    db_path = str(project_path / Path("chroma_db") / db_name)
    rag_chroma = RAG_Chroma(db_path)
    results = rag_chroma.select(keyword=keyword, limit=limit)
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
