"""
Command-line interface for Dolphin MCP.
"""

import asyncio
import sys
from .utils import parse_arguments
from .client import run_interaction

def main():
    """
    Main entry point for the CLI.
    """
    # Check for help flag first
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: dolphin-mcp-cli [--model <n>] [--quiet] [--config <file>] [--log-messages <file>] [--conversation-file <file>] 'your question'")
        print("\nOptions:")
        print("  --model <n>             Specify the model to use")
        print("  --quiet                 Suppress intermediate output")
        print("  --config <file>         Specify a custom config file (default: mcp_config.json)")
        print("  --log-messages <file>   Log all LLM interactions to a JSONL file")
        print("  --conversation-file <file>  Save and load conversation history to/from a JSONL file")
        print("  --help, -h              Show this help message")
        sys.exit(0)
        
    chosen_model_name, user_query, quiet_mode, config_path, log_messages_path, conversation_file = parse_arguments()
    if not user_query:
        print("Usage: dolphin-mcp-cli [--model <n>] [--quiet] [--config <file>] [--conversation-file <file>] 'your question'")
        sys.exit(1)

    # We do not pass a config object; we pass config_path
    final_text = asyncio.run(run_interaction(
        user_query=user_query,
        model_name=chosen_model_name,
        config_path=config_path,
        quiet_mode=quiet_mode,
        log_messages_path=log_messages_path,
        persist_conversation_to_file=conversation_file
    ))

    print("\n" + final_text.strip() + "\n")

if __name__ == "__main__":
    main()
