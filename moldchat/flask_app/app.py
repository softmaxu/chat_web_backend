from flask import Flask, request, jsonify
from tool.process_file import DocumentParser, TextCleaner
from pathlib import Path
import os

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
