# Testing Documentation

This document describes the comprehensive testing setup for the Wikipedia Agent project.

## Test Structure

The testing suite consists of multiple levels of testing to ensure comprehensive coverage:

### 1. Basic Tests (`test_basic.py`)
- **Purpose**: Test basic structure and parsing without API calls
- **Requirements**: No API key needed
- **Coverage**: Agent initialization, argument parsing, fallback logic

```bash
python test_basic.py
```

### 2. Smoke Tests (`test_smoke.py`)
- **Purpose**: Validate basic functionality without requiring API access
- **Requirements**: No API key needed
- **Coverage**: Module imports, file structure, error handling, dependencies

```bash
python test_smoke.py
```

### 3. Integration Tests (`test_integration.py`)
- **Purpose**: Test complete functionality with real API calls
- **Requirements**: GEMINI_API_KEY environment variable
- **Coverage**: Full workflow, Wikipedia integration, AI functionality

```bash
export GEMINI_API_KEY="your_api_key_here"
python test_integration.py
```

### 4. Documented Features Tests (`test_documented_features.py`)
- **Purpose**: Validate all features mentioned in README.md
- **Requirements**: GEMINI_API_KEY environment variable
- **Coverage**: All README examples, configuration options, performance

```bash
export GEMINI_API_KEY="your_api_key_here"
python test_documented_features.py
```

## GitHub Actions Integration

The project uses GitHub Actions for automated testing with the workflow file `.github/workflows/integration-tests.yml`.

### Workflow Structure

#### Job 1: `basic-tests`
- Runs on Python 3.8, 3.9, 3.10, 3.11
- Tests that don't require API key
- Validates basic functionality and error handling

#### Job 2: `integration-tests`
- Requires `basic-tests` to pass first
- Runs on Python 3.9, 3.11 (fewer versions for API-dependent tests)
- Uses `GEMINI_API_KEY` secret from repository settings
- Comprehensive functionality testing

### Test Categories in GitHub Actions

1. **Basic Structure Tests**
   - Module imports and structure
   - Argument parsing
   - Error handling without API key

2. **Integration Tests**
   - Complete functionality with real API
   - Wikipedia search and AI integration
   - CLI parameter validation

3. **Documentation Validation**
   - All README examples
   - Expected output format
   - Configuration options

4. **Performance & Reliability**
   - Response time validation
   - Multiple question sequences
   - Edge case handling

## Setting Up API Key

### For Local Testing
```bash
# Option 1: Environment variable
export GEMINI_API_KEY="your_api_key_here"

# Option 2: .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Option 3: Command line parameter
python wikipedia_agent.py --api-key "your_api_key_here" "your question"
```

### For GitHub Actions
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `GEMINI_API_KEY`
4. Value: Your actual Gemini API key
5. Click "Add secret"

## Test Coverage

### Functionality Tested
- ✅ Natural language question processing
- ✅ Wikipedia search integration
- ✅ Google Gemini AI integration
- ✅ Iterative search strategy
- ✅ Answer synthesis and formatting
- ✅ CLI parameter handling
- ✅ Error handling and validation
- ✅ Performance requirements
- ✅ All documented features

### Example Questions Tested
- "Who was the first person to walk on the moon?"
- "What is the capital of France?"
- "How does photosynthesis work?"
- "Who invented the telephone?"
- "What is the capital of Japan?"
- "When was the Internet created?"
- "What are the benefits of renewable energy?"

### Configuration Options Tested
- `--max-iterations` parameter
- `--api-key` parameter
- Environment variable configuration
- Combined parameter usage

## Running Tests Locally

### Prerequisites
```bash
pip install -r requirements.txt
```

### Test Sequence
1. **Start with smoke tests** (no API key needed):
   ```bash
   python test_smoke.py
   ```

2. **Run basic tests** (no API key needed):
   ```bash
   python test_basic.py
   ```

3. **Set up API key** and run integration tests:
   ```bash
   export GEMINI_API_KEY="your_api_key"
   python test_integration.py
   ```

4. **Test documented features**:
   ```bash
   python test_documented_features.py
   ```

### Expected Output
All tests should show:
- ✅ Individual test results
- 📊 Summary statistics
- 🎉 Success message if all tests pass

## Troubleshooting

### Common Issues
1. **Missing API Key**: Ensure GEMINI_API_KEY is set correctly
2. **Network Issues**: Tests may timeout if internet connection is slow
3. **Rate Limiting**: Add delays between tests if hitting API limits
4. **Dependencies**: Run `pip install -r requirements.txt` if imports fail

### Test Timeouts
- Most tests have 60-120 second timeouts
- If tests consistently timeout, check network connectivity
- API rate limits may cause temporary failures

## Adding New Tests

### For New Features
1. Add unit tests to `test_basic.py` for structure
2. Add integration tests to `test_integration.py` for functionality
3. Update `test_documented_features.py` if feature is documented
4. Update this TESTING.md file

### Test Guidelines
- Use descriptive test names
- Include both success and failure scenarios
- Add appropriate timeouts for API-dependent tests
- Validate both output content and format
- Include performance considerations