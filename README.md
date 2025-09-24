# Wikipedia Agent

A simple command line agent to answer natural language questions using Wikipedia as a knowledge source and Google's Gemini 2.0 Flash model for intelligent search term generation and answer synthesis.

## Features

- **Natural Language Questions**: Ask questions in plain English
- **Intelligent Search**: Uses Google Gemini 2.0 Flash to generate relevant Wikipedia search terms
- **Iterative Search**: Automatically tries multiple search strategies until it finds a satisfactory answer
- **Simple CLI**: Easy-to-use command line interface
- **No Complex Frameworks**: Built with minimal dependencies for maximum reliability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rvpanchani/wikipedia-agent.git
cd wikipedia-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get a Google Gemini API key:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key

4. Set up your API key (choose one method):

   **Option A: Environment variable**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

   **Option B: .env file**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

   **Option C: Command line argument**
   ```bash
   python wikipedia_agent.py --api-key "your_api_key_here" "your question"
   ```

## Usage

### Basic Usage

```bash
python wikipedia_agent.py "Who was the first person to walk on the moon?"
```

### Advanced Usage

```bash
# Increase search iterations for complex questions
python wikipedia_agent.py --max-iterations 5 "How does photosynthesis work in detail?"

# Use API key from command line
python wikipedia_agent.py --api-key "your_key" "What is quantum computing?"
```

### Example Output

```
🤔 Processing question: Who was the first person to walk on the moon?

📍 Iteration 1/3
🔍 Generated search terms: Neil Armstrong, Apollo 11, Moon landing, First moonwalk, Lunar surface
   Searching Wikipedia for: Neil Armstrong
   ✅ Found content (2847 characters)

============================================================
📝 ANSWER:
============================================================
Neil Armstrong was the first person to walk on the moon. He was an American astronaut and aeronautical engineer who became the first person to step onto the lunar surface on July 20, 1969, during the Apollo 11 mission. Armstrong famously said "That's one small step for man, one giant leap for mankind" as he stepped onto the Moon's surface.

🔍 Search terms used: Neil Armstrong
```

## How It Works

1. **Question Processing**: The agent receives your natural language question
2. **Search Term Generation**: Uses Google Gemini 2.0 Flash to generate relevant Wikipedia search terms
3. **Wikipedia Search**: Searches Wikipedia for each generated term
4. **Content Analysis**: Retrieves and analyzes Wikipedia content
5. **Answer Generation**: Uses Gemini to synthesize a comprehensive answer from the Wikipedia content
6. **Iterative Refinement**: If the answer isn't satisfactory, tries different search terms (up to max iterations)

## Configuration Options

- `--max-iterations`: Number of search iterations (default: 3)
- `--api-key`: Google Gemini API key (overrides environment variable)

## Dependencies

- `wikipedia==1.4.0`: Wikipedia API access
- `google-generativeai==0.8.3`: Google Gemini API client
- `python-dotenv==1.0.1`: Environment variable management

## Requirements

- Python 3.7 or higher
- Google Gemini API key
- Internet connection

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Gemini API key is correctly set
2. **Network Issues**: Ensure you have a stable internet connection
3. **Wikipedia Disambiguation**: If Wikipedia returns multiple pages for a term, the agent will try the first option

### Getting Help

```bash
python wikipedia_agent.py --help
```

## Examples of Questions You Can Ask

- "Who invented the telephone?"
- "What caused World War I?"
- "How does the human heart work?"
- "What is the capital of Japan?"
- "When was the Internet created?"
- "What are the benefits of renewable energy?"

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
