#!/usr/bin/env python

import os
import sys
import asyncio

# Add the src directory to the Python path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import from the package
from dolphin_mcp.utils import parse_arguments
from dolphin_mcp.client import run_interaction

async def cli_main():
    """
    Main entry point for the CLI script.
    """
    chosen_model_name, user_query, quiet_mode, config_path, log_messages_path, conversation_file = parse_arguments()
    if not user_query:
        print("Usage: python dolphin_mcp.py [--model <n>] [--quiet] [--config <file>] [--log-messages <file>] [--conversation-file <file>] 'your question'")
        sys.exit(1)

    # Pass through to the package implementation
    final_text = await run_interaction(
        user_query=user_query,
        model_name=chosen_model_name,
        config_path=config_path,
        quiet_mode=quiet_mode,
        log_messages_path=log_messages_path,
        persist_conversation_to_file=conversation_file
    )

    print("\n" + final_text.strip() + "\n")

def main():
    """
    Entry point for the script.
    """
    asyncio.run(cli_main())

if __name__ == "__main__":
    main()
