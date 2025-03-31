<<<<<<< HEAD
# tdsproject2
TDS_project_2
=======
# Automated Question Answering API

An LLM-powered API for automatically answering questions from course materials.

## Project Description

This FastAPI application uses advanced language models to provide accurate answers to questions. The system can:

1. Process natural language questions and find similar questions in its database
2. Generate and execute Python or Bash code to solve computational problems
3. Handle file uploads (ZIP, CSV) and extract relevant information

## Features

### Question Processing
- Determines whether code execution is needed or if a direct answer is sufficient

### Code Generation and Execution
- Generates Python or Bash code based on the question
- Executes code in a secure temporary environment
- Extracts code from markdown code blocks if necessary

### File Processing
- Handles ZIP files by extracting contents
- Processes CSV files to extract data
- Provides file context to the LLM for better answers

### API Integration
- Connects to LLM API for generating answers and code
- Uses environment variables for secure API token storage

## Setup

### Prerequisites
- Python 3.8+
- FastAPI
- scikit-learn
- dotenv

### Installation

1. Clone the repository:
```bash
git clone https://github.com/krishna-gramener/tdsproject2.git
cd tdsproject2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API token:
```
GITHUB_API_KEY=your_api_token_here
VERCEL_API_KEY=your_api_token_here
```

4. Run the application:
```bash
uvicorn app:app --reload
```

## API Usage

### Endpoint: `/api`

**Method**: POST

**Parameters**:
- `question` (required): The question to answer
- `file` (optional): A file to upload (ZIP or CSV)

### Example Request

```bash
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: multipart/form-data" \
  -F "question=What is the output of pd.read_csv('data.csv').head()?" \
  -F "file=@data.csv"
```

## Error Handling

The API returns detailed error messages when:
- Code execution fails
- The LLM API returns an error
- File processing encounters issues

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

MIT
>>>>>>> 4bd6185 (Initial commit)
