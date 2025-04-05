"""
OpenAI provider implementation for Dolphin MCP.
"""

import os
import json
import time
import uuid
from typing import Dict, List, Any, AsyncGenerator, Optional, Union

from openai import AsyncOpenAI, APIError, RateLimitError

# Get rate limit from env var (in seconds) or default to 60 seconds (1 minute)
def get_rate_limit_seconds():
    try:
        return float(os.getenv("OPENAI_RATE_LIMIT_SECONDS", "1"))
    except (ValueError, TypeError):
        return 1.0


async def generate_with_openai_stream(client: AsyncOpenAI, model_name: str, conversation: List[Dict],
                                    formatted_functions: List[Dict], temperature: Optional[float] = None,
                                    top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> AsyncGenerator:
    """Internal function for streaming generation"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=conversation,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=[{"type": "function", "function": f} for f in formatted_functions],
            tool_choice="auto",
            stream=True
        )

        current_tool_calls = []
        current_content = ""

        async for chunk in response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                # Immediately yield each token without buffering
                yield {"assistant_text": delta.content, "tool_calls": [], "is_chunk": True, "token": True}
                current_content += delta.content

            # Handle tool call updates
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    # Initialize or update tool call
                    while tool_call.index >= len(current_tool_calls):
                        current_tool_calls.append({
                            "id": "",
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        })
                    
                    current_tool = current_tool_calls[tool_call.index]
                    
                    # Update tool call properties
                    if tool_call.id:
                        current_tool["id"] = tool_call.id
                    
                    if tool_call.function.name:
                        current_tool["function"]["name"] = (
                            current_tool["function"]["name"] + tool_call.function.name
                        )
                    
                    if tool_call.function.arguments:
                        # Properly accumulate JSON arguments
                        current_args = current_tool["function"]["arguments"]
                        new_args = tool_call.function.arguments
                        
                        # Handle special cases for JSON accumulation
                        if new_args.startswith("{") and not current_args:
                            current_tool["function"]["arguments"] = new_args
                        elif new_args.endswith("}") and current_args:
                            # If we're receiving the end of the JSON object
                            if not current_args.endswith("}"):
                                current_tool["function"]["arguments"] = current_args + new_args
                        else:
                            # Middle part of JSON - append carefully
                            current_tool["function"]["arguments"] += new_args

            # If this is the last chunk, yield final state with complete tool calls
            if chunk.choices[0].finish_reason is not None:
                # Clean up and validate tool calls
                final_tool_calls = []
                for tc in current_tool_calls:
                    if tc["id"] and tc["function"]["name"]:
                        try:
                            # Ensure arguments is valid JSON
                            args = tc["function"]["arguments"].strip()
                            if not args or args.isspace():
                                args = "{}"
                            # Parse and validate JSON
                            parsed_args = json.loads(args)
                            tc["function"]["arguments"] = json.dumps(parsed_args)
                            # Ensure ID exists
                            if not tc.get("id"):
                                tc["id"] = f"call_{uuid.uuid4()}"
                            final_tool_calls.append(tc)
                        except json.JSONDecodeError:
                            # If arguments are malformed, try to fix common issues
                            args = tc["function"]["arguments"].strip()
                            # Remove any trailing commas
                            args = args.rstrip(",")
                            # Ensure proper JSON structure
                            if not args.startswith("{"):
                                args = "{" + args
                            if not args.endswith("}"):
                                args = args + "}"
                            try:
                                # Try parsing again after fixes
                                parsed_args = json.loads(args)
                                tc["function"]["arguments"] = json.dumps(parsed_args)
                                # Ensure ID exists
                                if not tc.get("id"):
                                    tc["id"] = f"call_{uuid.uuid4()}"
                                final_tool_calls.append(tc)
                            except json.JSONDecodeError:
                                # If still invalid, default to empty object
                                tc["function"]["arguments"] = "{}"
                                # Ensure ID exists
                                if not tc.get("id"):
                                    tc["id"] = f"call_{uuid.uuid4()}"
                                final_tool_calls.append(tc)

                yield {
                    "assistant_text": current_content,
                    "tool_calls": final_tool_calls,
                    "is_chunk": False
                }

    except Exception as e:
        yield {"assistant_text": f"OpenAI error: {str(e)}", "tool_calls": [], "is_chunk": False}

async def generate_with_openai_sync(client: AsyncOpenAI, model_name: str, conversation: List[Dict], 
                                  formatted_functions: List[Dict], temperature: Optional[float] = None,
                                  top_p: Optional[float] = None, max_tokens: Optional[int] = None, num_retries: int = 3) -> Dict:
    """Internal function for non-streaming generation"""
    try:
        # print(f"Generating with OpenAI: {model_name}\n{conversation}\n{formatted_functions}\n{temperature}\n{top_p}\n{max_tokens}")
        response = await client.chat.completions.create(
            extra_body={},
            model=model_name,
            messages=conversation,
            temperature=temperature,
            top_p=top_p,
            max_tokens=32000, # max_tokens,
            tools=[{"type": "function", "function": f} for f in formatted_functions],
            tool_choice="auto",
            stream=False
        )
        
        # Log the raw response for debugging
        # print(f"Raw OpenAI Response: {response.model_dump_json(indent=2)}")

        if not response.choices:
            time.sleep(get_rate_limit_seconds())
            if num_retries > 0:
                return await generate_with_openai_sync(client, model_name, conversation, formatted_functions, temperature, top_p, max_tokens, num_retries - 1)
            else:
                return {"assistant_text": "OpenAI error: No choices returned in the response.", "tool_calls": []}

        choice = response.choices[0]

        if choice.message is None:
            # Handle cases where the message object itself might be missing
            finish_reason = choice.finish_reason or "unknown"
            return {"assistant_text": f"OpenAI error: No message found in the response choice. Finish reason: {finish_reason}", "tool_calls": []}

        assistant_text = choice.message.content or ""
        tool_calls = []

        # Use getattr for safer access to tool_calls, check it's not None
        message_tool_calls = getattr(choice.message, 'tool_calls', None)
        if message_tool_calls:
            print(f"\n{assistant_text}\n")
            for tc in message_tool_calls:
                if tc.type == 'function':
                    # Ensure function and its attributes exist
                    if tc.function and tc.function.name:
                        tool_call = {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                # Provide default empty string if arguments is None
                                "arguments": tc.function.arguments or "{}" 
                            }
                        }
                        # Ensure arguments is valid JSON
                        try:
                            json.loads(tool_call["function"]["arguments"])
                        except (json.JSONDecodeError, TypeError): # Added TypeError for safety
                            # Attempt to fix common issues or default to empty object
                            args_str = str(tool_call["function"]["arguments"]).strip()
                            if not args_str or args_str.isspace():
                                tool_call["function"]["arguments"] = "{}"
                            else:
                                # Minimal fix attempt (e.g., wrap if not bracketed)
                                if not args_str.startswith("{") or not args_str.endswith("}"):
                                     args_str = f"{{{args_str}}}" # Basic wrapping, might not always work
                                try:
                                    json.loads(args_str)
                                    tool_call["function"]["arguments"] = args_str
                                except json.JSONDecodeError:
                                     tool_call["function"]["arguments"] = "{}" # Fallback
                    tool_calls.append(tool_call)

            # Ensure all tool calls have an ID before returning
            for tc in tool_calls:
                if not tc.get("id"):
                    tc["id"] = f"call_{uuid.uuid4()}"

        return {"assistant_text": assistant_text, "tool_calls": tool_calls}

    except APIError as e:
        print(f"OpenAI API Error encountered: Status Code: {e.status_code}, Body: {e.body}")
        return {"assistant_text": f"OpenAI API error: {str(e)}", "tool_calls": []}
    except RateLimitError as e:
        print(f"OpenAI Rate Limit Error encountered: {e}")
        return {"assistant_text": f"OpenAI rate limit: {str(e)}", "tool_calls": []}
    except Exception as e:
        # Add more context to the generic exception log
        response_summary = "No response object available"
        try:
            # Try to safely get some info from the response if it exists
            if 'response' in locals() and response:
                 response_summary = f"Response ID: {getattr(response, 'id', 'N/A')}, Model: {getattr(response, 'model', 'N/A')}, Choices count: {len(response.choices) if response.choices else 0}"
                 if response.choices and response.choices[0]:
                     response_summary += f", Finish Reason: {getattr(response.choices[0], 'finish_reason', 'N/A')}"
        except Exception as inner_e:
            response_summary = f"Could not summarize response: {inner_e}"
            
        print(f"Unexpected error during OpenAI sync generation: {e.__class__.__name__}: {str(e)}. Response summary: {response_summary}")
        import traceback
        traceback.print_exc() # Print traceback for detailed debugging
        return {"assistant_text": f"Unexpected OpenAI error: {e.__class__.__name__}: {str(e)}", "tool_calls": []}

async def generate_with_openai(conversation: List[Dict], model_cfg: Dict, 
                             all_functions: List[Dict], stream: bool = False) -> Union[Dict, AsyncGenerator]:
    """
    Generate text using OpenAI's API.
    
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call
        stream: Whether to stream the response
        
    Returns:
        If stream=False: Dict containing assistant_text and tool_calls
        If stream=True: AsyncGenerator yielding chunks of assistant text and tool calls
    """
    api_key = model_cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
    if "apiBase" in model_cfg:
        client = AsyncOpenAI(api_key=api_key, base_url=model_cfg["apiBase"])
    else:
        client = AsyncOpenAI(api_key=api_key)

    model_name = model_cfg["model"]
    temperature = model_cfg.get("temperature", None)
    top_p = model_cfg.get("top_p", None)
    max_tokens = model_cfg.get("max_tokens", None)

    # Format functions for OpenAI API
    formatted_functions = []
    for func in all_functions:
        formatted_func = {
            "name": func["name"],
            "description": func["description"],
            "parameters": func["parameters"]
        }
        formatted_functions.append(formatted_func)
        
    # Format conversation for OpenAI API format
    formatted_conversation = []
    for msg in conversation:
        if "role" not in msg:
            # Skip malformed messages
            continue
            
        if msg["role"] == "tool":
            # Format tool response messages
            if "tool_call_id" in msg and "name" in msg and "content" in msg:
                formatted_conversation.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "name": msg.get("name", ""),
                    "content": msg.get("content", "")
                })
        elif msg["role"] == "assistant":
            # Format assistant messages that might have tool calls
            formatted_msg = {"role": "assistant"}
            
            # Set content (must be present even if empty)
            formatted_msg["content"] = msg.get("content", "")
            
            # Handle tool calls if present
            if "tool_calls" in msg and msg["tool_calls"]:
                # Format tool calls correctly for OpenAI API
                formatted_tool_calls = []
                for tc in msg["tool_calls"]:
                    if "id" in tc and "function" in tc and "name" in tc["function"]:
                        formatted_tool_call = {
                            "id": tc["id"],
                            "type": "function",  # This was missing
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"].get("arguments", "{}")
                            }
                        }
                        formatted_tool_calls.append(formatted_tool_call)
                
                if formatted_tool_calls:
                    formatted_msg["tool_calls"] = formatted_tool_calls
            
            formatted_conversation.append(formatted_msg)
        else:
            # User or system messages
            formatted_conversation.append({
                "role": msg["role"],
                "content": msg.get("content", "")
            })

    if stream:
        return generate_with_openai_stream(
            client, model_name, formatted_conversation, formatted_functions,
            temperature, top_p, max_tokens
        )
    else:
        return await generate_with_openai_sync(
            client, model_name, formatted_conversation, formatted_functions,
            temperature, top_p, max_tokens
        )
