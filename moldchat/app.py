from flask import Flask, request, jsonify
from extract_text import DocumentParser
import os

app = Flask(__name__)

@app.route('/process_file', methods=['POST'])
def process_file():
    data = request.get_json()
    file_path = data.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400

    document_parser = DocumentParser(file_path)
    document_parser.extract_text()
    result = f"Processing file: {file_path}"
    return jsonify({"message": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
