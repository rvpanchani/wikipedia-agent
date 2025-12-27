#!/usr/bin/env python3
"""
Gradio UI for Wikipedia Agent.

Provides a web-based interface for:
- Wikipedia Agent V2: Question answering with Wikipedia search and code execution
- Image Analysis: Vision-based image analysis using LLMs
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Generator

import gradio as gr
from dotenv import load_dotenv

from wikipedia_agent.v2.agent import WikipediaAgentV2
from wikipedia_agent.tools.image_analyser import ImageAnalyzerAgent
from wikipedia_agent.core.providers import ProviderFactory

# Load environment variables
load_dotenv()

# Provider configurations
AGENT_PROVIDERS = ["auto", "openai", "azure", "gemini", "ollama", "huggingface"]
IMAGE_PROVIDERS = ["openai", "huggingface"]

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-3.5-turbo",
    "azure": "gpt-35-turbo",
    "gemini": "gemini-2.0-flash",
    "ollama": "qwen3:0.6b",
    "huggingface": "microsoft/DialoGPT-medium",
}

IMAGE_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "huggingface": "qwen/qwen3-vl-4b",
}


class VerboseCapture:
    """Capture verbose output from agent for streaming to UI."""
    
    def __init__(self):
        self.messages: list[str] = []
        self._original_stdout = None
    
    def capture_log(self, message: str) -> None:
        """Capture a log message."""
        self.messages.append(message)
    
    def get_output(self) -> str:
        """Get all captured output as a string."""
        return "\n".join(self.messages)
    
    def clear(self) -> None:
        """Clear captured messages."""
        self.messages = []


def get_detected_provider() -> str:
    """Auto-detect the configured provider from environment variables."""
    detected = ProviderFactory.auto_detect_provider()
    return detected if detected else "ollama"


def query_wikipedia_agent(
    question: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_search_attempts: int,
    api_key: str,
    enable_code_execution: bool,
    verbose: bool,
    azure_endpoint: str,
    azure_api_version: str,
    ollama_base_url: str,
) -> Generator[Tuple[str, str], None, None]:
    """
    Query the Wikipedia Agent V2 with the given parameters.
    
    Yields tuples of (thinking_output, answer_output) for streaming updates.
    Returns formatted output including answer, sources, and executed code.
    """
    if not question.strip():
        yield ("", "⚠️ Please enter a question.")
        return
    
    # Handle provider selection
    provider_name = None if provider == "auto" else provider
    
    # Build provider kwargs
    provider_kwargs = {}
    if azure_endpoint and azure_endpoint.strip():
        provider_kwargs["azure_endpoint"] = azure_endpoint.strip()
    if azure_api_version and azure_api_version.strip():
        provider_kwargs["api_version"] = azure_api_version.strip()
    if ollama_base_url and ollama_base_url.strip():
        provider_kwargs["base_url"] = ollama_base_url.strip()
    
    # Use API key from input or fall back to env
    effective_api_key = api_key.strip() if api_key and api_key.strip() else None
    
    # Use model from input or fall back to provider default
    effective_model = model.strip() if model and model.strip() else None
    
    try:
        # Create verbose capture
        verbose_capture = VerboseCapture()
        
        agent = WikipediaAgentV2(
            provider_name=provider_name,
            api_key=effective_api_key,
            model=effective_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_search_attempts=max_search_attempts,
            enable_code_execution=enable_code_execution,
            verbose=True,  # Always enable verbose internally to capture logs
            **provider_kwargs
        )
        
        if verbose:
            # Show initial thinking state
            yield ("🔄 *Processing query...*", "")
        
        # Patch the agent's _log method to capture verbose output
        # Override the _log method to capture output and optionally print to console
        def patched_log(message: str) -> None:
            verbose_capture.capture_log(message)
            # Only print to console if user enabled verbose mode
            if verbose:
                print(message)
        
        agent._log = patched_log
        
        # Run the query (this will now populate verbose_capture via patched _log)
        response = agent.query(question)
        
        # Get the final thinking output
        thinking_output = ""
        if verbose:
            thinking_output = verbose_capture.get_output()
            if thinking_output:
                # Format thinking output nicely as a code block for better readability
                thinking_output = "### 🧠 Agent Thinking Process\n\n```\n" + thinking_output + "\n```"
            else:
                thinking_output = "*No verbose output captured.*"
        
        # Format the answer with markdown support
        formatted_answer = _format_answer_as_markdown(response)
        
        yield (thinking_output, formatted_answer)
        
    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}\n\n💡 **Tip:** Check your provider configuration and API keys."
        yield ("", error_msg)


def _format_answer_as_markdown(response) -> str:
    """Format the agent response as markdown for rich display."""
    output_parts = []
    
    # Main answer section
    output_parts.append("## 📝 Answer\n")
    output_parts.append(response.answer)
    
    # Code section (if executed)
    if response.code_executed:
        output_parts.append("\n---\n")
        output_parts.append("### 🔢 Code Executed\n")
        output_parts.append(f"```python\n{response.code_executed}\n```")
        
        if response.code_result:
            output_parts.append(f"\n**📊 Result:** `{response.code_result}`")
    
    # Sources section
    if response.sources:
        output_parts.append("\n---\n")
        output_parts.append("### 📚 Sources\n")
        for i, url in enumerate(response.sources, 1):
            output_parts.append(f"{i}. [{url}]({url})")
    
    return "\n".join(output_parts)


def analyze_image(
    image,
    prompt: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> str:
    """
    Analyze an image using the Image Analyzer Agent.
    
    Returns the analysis result.
    """
    if image is None:
        return "⚠️ Please upload an image."
    
    if not prompt.strip():
        prompt = "Describe this image in detail."
    
    # Use API key from input or fall back to env
    effective_api_key = api_key.strip() if api_key and api_key.strip() else None
    
    # Use model from input or fall back to provider default
    effective_model = model.strip() if model and model.strip() else None
    
    # Use base_url from input or fall back to default
    effective_base_url = base_url.strip() if base_url and base_url.strip() else None
    
    try:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = Path(tmp.name)
        
        try:
            analyzer = ImageAnalyzerAgent(
                provider_name=provider,
                api_key=effective_api_key,
                model=effective_model,
                base_url=effective_base_url,
                temperature=temperature,
            )
            
            result = analyzer.analyze_image(image_path, prompt)
            return result
            
        finally:
            # Clean up temp file
            if image_path.exists():
                image_path.unlink()
                
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n💡 Tip: Image analysis requires an OpenAI-compatible vision model."


def update_model_placeholder(provider: str) -> gr.update:
    """Update model input placeholder based on selected provider."""
    if provider == "auto":
        return gr.update(placeholder="Auto-detected based on provider")
    default = DEFAULT_MODELS.get(provider, "")
    return gr.update(placeholder=f"Default: {default}")


def update_image_model_placeholder(provider: str) -> gr.update:
    """Update image model input placeholder based on selected provider."""
    default = IMAGE_DEFAULT_MODELS.get(provider, "")
    return gr.update(placeholder=f"Default: {default}")


def update_provider_fields(provider: str) -> Tuple[gr.update, gr.update, gr.update]:
    """Show/hide provider-specific fields based on selection."""
    is_azure = provider == "azure"
    is_ollama = provider == "ollama"
    
    return (
        gr.update(visible=is_azure),  # Azure endpoint
        gr.update(visible=is_azure),  # Azure API version
        gr.update(visible=is_ollama), # Ollama base URL
    )


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    
    detected_provider = get_detected_provider()
    
    with gr.Blocks(
        title="Wikipedia Agent",
    ) as demo:
        gr.Markdown(
            """
            # 🌐 Wikipedia Agent
            
            An AI-powered research assistant that answers questions using Wikipedia and executes code for calculations.
            """
        )
        
        with gr.Tabs():
            # ===== WIKIPEDIA AGENT TAB =====
            with gr.TabItem("📚 Wikipedia Agent", id="agent"):
                gr.Markdown(
                    """
                    Ask any question! The agent will:
                    - Search Wikipedia for relevant information
                    - Execute Python code for calculations
                    - Provide sources and citations
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Question",
                            placeholder="e.g., What is the speed of light in km/h? Calculate it.",
                            lines=3,
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("🔍 Ask", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ Clear", scale=1)
                        
                        # Thinking/Verbose output panel (collapsible accordion)
                        with gr.Accordion("🧠 Thinking Process", open=False, visible=True) as thinking_accordion:
                            thinking_output = gr.Markdown(
                                value="*Verbose mode will show the agent's reasoning process here...*",
                                elem_id="thinking-output",
                            )
                        
                        # Main response with Markdown support
                        agent_output = gr.Markdown(
                            value="",
                            label="Response",
                            elem_id="agent-response",
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        
                        provider_dropdown = gr.Dropdown(
                            choices=AGENT_PROVIDERS,
                            value="auto",
                            label="Provider",
                            info=f"Auto-detected: {detected_provider}",
                        )
                        
                        model_input = gr.Textbox(
                            label="Model",
                            placeholder="Auto-detected based on provider",
                            info="Leave empty for provider default",
                        )
                        
                        api_key_input = gr.Textbox(
                            label="API Key",
                            placeholder="Uses environment variable if empty",
                            type="password",
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            temperature_slider = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature",
                                info="Higher = more creative, Lower = more focused",
                            )
                            
                            max_tokens_slider = gr.Slider(
                                minimum=100,
                                maximum=4000,
                                value=1000,
                                step=100,
                                label="Max Tokens",
                            )
                            
                            max_search_slider = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=2,
                                step=1,
                                label="Max Search Attempts",
                            )
                            
                            code_execution_checkbox = gr.Checkbox(
                                value=True,
                                label="Enable Code Execution",
                                info="Allow agent to run Python code for calculations",
                            )
                            
                            verbose_checkbox = gr.Checkbox(
                                value=False,
                                label="Verbose Mode",
                                info="Show agent's thinking process in the UI",
                            )
                        
                        with gr.Accordion("Azure Settings", open=False, visible=False) as azure_accordion:
                            azure_endpoint_input = gr.Textbox(
                                label="Azure Endpoint",
                                placeholder="https://your-resource.openai.azure.com/",
                            )
                            
                            azure_api_version_input = gr.Textbox(
                                label="Azure API Version",
                                placeholder="2024-02-01",
                            )
                        
                        with gr.Accordion("Ollama Settings", open=False, visible=False) as ollama_accordion:
                            ollama_base_url_input = gr.Textbox(
                                label="Ollama Base URL",
                                placeholder="http://localhost:11434",
                                value="",
                            )
                
                # Event handlers for Wikipedia Agent
                provider_dropdown.change(
                    fn=update_model_placeholder,
                    inputs=[provider_dropdown],
                    outputs=[model_input],
                )
                
                provider_dropdown.change(
                    fn=update_provider_fields,
                    inputs=[provider_dropdown],
                    outputs=[azure_accordion, azure_api_version_input, ollama_accordion],
                )
                
                submit_btn.click(
                    fn=query_wikipedia_agent,
                    inputs=[
                        question_input,
                        provider_dropdown,
                        model_input,
                        temperature_slider,
                        max_tokens_slider,
                        max_search_slider,
                        api_key_input,
                        code_execution_checkbox,
                        verbose_checkbox,
                        azure_endpoint_input,
                        azure_api_version_input,
                        ollama_base_url_input,
                    ],
                    outputs=[thinking_output, agent_output],
                )
                
                question_input.submit(
                    fn=query_wikipedia_agent,
                    inputs=[
                        question_input,
                        provider_dropdown,
                        model_input,
                        temperature_slider,
                        max_tokens_slider,
                        max_search_slider,
                        api_key_input,
                        code_execution_checkbox,
                        verbose_checkbox,
                        azure_endpoint_input,
                        azure_api_version_input,
                        ollama_base_url_input,
                    ],
                    outputs=[thinking_output, agent_output],
                )
                
                clear_btn.click(
                    fn=lambda: ("", "*Verbose mode will show the agent's reasoning process here...*", ""),
                    outputs=[question_input, thinking_output, agent_output],
                )
            
            # ===== IMAGE ANALYSIS TAB =====
            with gr.TabItem("🖼️ Image Analysis", id="image"):
                gr.Markdown(
                    """
                    Upload an image and ask questions about it using vision-capable LLMs.
                    
                    **Note:** Requires an OpenAI-compatible vision model (e.g., GPT-4o, Qwen-VL).
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                        )
                        
                        image_prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe this image in detail.",
                            lines=2,
                        )
                        
                        with gr.Row():
                            image_submit_btn = gr.Button("🔍 Analyze", variant="primary", scale=2)
                            image_clear_btn = gr.Button("🗑️ Clear", scale=1)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        
                        image_provider_dropdown = gr.Dropdown(
                            choices=IMAGE_PROVIDERS,
                            value="openai",
                            label="Provider",
                            info="Must be OpenAI-compatible with vision support",
                        )
                        
                        image_model_input = gr.Textbox(
                            label="Model",
                            placeholder=f"Default: {IMAGE_DEFAULT_MODELS['openai']}",
                            info="Leave empty for provider default",
                        )
                        
                        image_api_key_input = gr.Textbox(
                            label="API Key",
                            placeholder="Uses IMAGE_ANALYSER_API_KEY env var if empty",
                            type="password",
                        )
                        
                        image_base_url_input = gr.Textbox(
                            label="Base URL",
                            placeholder="http://localhost:1234/v1/",
                            info="OpenAI-compatible endpoint URL",
                        )
                        
                        image_temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                        )
                
                image_output = gr.Textbox(
                    label="Analysis Result",
                    lines=10,
                )
                
                # Event handlers for Image Analysis
                image_provider_dropdown.change(
                    fn=update_image_model_placeholder,
                    inputs=[image_provider_dropdown],
                    outputs=[image_model_input],
                )
                
                image_submit_btn.click(
                    fn=analyze_image,
                    inputs=[
                        image_input,
                        image_prompt_input,
                        image_provider_dropdown,
                        image_model_input,
                        image_api_key_input,
                        image_base_url_input,
                        image_temperature_slider,
                    ],
                    outputs=[image_output],
                )
                
                image_clear_btn.click(
                    fn=lambda: (None, "", ""),
                    outputs=[image_input, image_prompt_input, image_output],
                )
        
        gr.Markdown(
            """
            ---
            **Tips:**
            - For calculations, the agent will automatically execute Python code
            - Check the **Sources** section for Wikipedia references
            - Enable **Verbose Mode** to see the agent's thinking process in the UI
            - Responses are rendered with **Markdown** for better formatting
            """
        )
    
    return demo


def launch_ui(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    **kwargs,
) -> None:
    """
    Launch the Gradio UI.
    
    Args:
        share: Create a public shareable link
        server_name: Server hostname (use "0.0.0.0" for all interfaces)
        server_port: Server port number
        **kwargs: Additional arguments passed to demo.launch()
    """
    demo = create_demo()
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        **kwargs,
    )


if __name__ == "__main__":
    launch_ui()
