async def stream_results(result):
    """
    Stream and display all events from agent execution.
    Handles: Web Search, Code Interpreter, Function calls (MCP tools), File Search, Reasoning, and Text output.
    """
    async for ev in result.stream_events():
        # Skip agent updates silently
        if ev.type == "agent_updated_stream_event":
            continue
        
        # Handle raw response events
        if ev.type == "raw_response_event":
            data = ev.data
            event_type = getattr(data, 'type', None)
            
            # 1. Tool call initiated
            if event_type == "response.output_item.added":
                item = data.item
                item_type = getattr(item, 'type', None)
                
                # Web search started
                if item_type == 'web_search_call':
                    action = getattr(item, 'action', None)
                    query = getattr(action, 'query', None) if action else None
                    print(f"🌐 [Web Search] {query if query else 'searching...'}")
                    print('===' * 20)
                
                # Code interpreter started
                elif item_type == 'code_interpreter_call':
                    print(f"🐍 [Code Interpreter]")
                    print('===' * 20)
                
                # Function call started (these are your MCP tools!)
                elif item_type == 'function_call':
                    name = getattr(item, 'name', 'unknown')
                    print(f"⚙️  [Function] {name}")
                    print('===' * 20)
                
                # File search started
                elif item_type == 'file_search_call':
                    queries = getattr(item, 'queries', [])
                    if queries:
                        print(f"📁 [File Search]")
                        for query in queries:
                            print(f"   → Query: {query}")
                    else:
                        print(f"📁 [File Search] searching...")
                    print('===' * 20)
                    print('\n')
            
            # 2. Function arguments completed
            elif event_type == "response.function_call_arguments.done":
                args = getattr(data, 'arguments', '')
                print(f"   → Args: {args}\n")
            
            # 3. Code interpreter code done
            elif event_type == "response.code_interpreter_call_code.done":
                code = getattr(data, 'code', '')
                if code:
                    print(f"   → Code: {code}\n")
            
            # 4. Tool/item completion
            elif event_type == "response.output_item.done":
                item = data.item
                item_type = getattr(item, 'type', None)
                
                # Web search completed
                if item_type == 'web_search_call':
                    status = getattr(item, 'status', None)
                    if status == 'completed':
                        print(f"✅ [Web Search Complete]")
                        print('===' * 20)
                        print('\n')
                
                # Code interpreter completed
                elif item_type == 'code_interpreter_call':
                    status = getattr(item, 'status', None)
                    if status == 'completed':
                        print(f"✅ [Code Interpreter Complete]")
                        print('===' * 20)
                        print('\n')
                
                # Function call completed (MCP tools)
                elif item_type == 'function_call':
                    status = getattr(item, 'status', None)
                    if status == 'completed':
                        name = getattr(item, 'name', 'unknown')
                        print(f"✅ [Function Complete] {name}")
                        print('===' * 20)
                        print('\n')
                
                # File search completed
                elif item_type == 'file_search_call':
                    status = getattr(item, 'status', None)
                    if status == 'completed':
                        print(f"✅ [File Search Complete]")
                        print('===' * 20)
                        print('\n')
                
                # Reasoning completed
                elif item_type == 'reasoning':
                    print(f"🧠 [Reasoning Complete]")
                    if hasattr(item, 'summary') and item.summary:
                        summary_text = '\n'.join([t.text for t in item.summary])
                        print(f"{summary_text}")
                    print('===' * 20)
                    print('\n')
            
            # 5. Text deltas (streaming assistant response)
            elif event_type == "response.output_text.delta":
                delta = getattr(data, 'delta', '')
                if delta:
                    print(delta, end='', flush=True)
            
            # 6. Text done (full text available)
            elif event_type == "response.output_text.done":
                # Text streaming completed, just add newline
                pass
            
            # 7. Reasoning summary
            elif event_type == "response.reasoning_summary_text.done":
                text = getattr(data, 'text', '')
                if text:
                    print(f"\n📝 [Reasoning Summary]")
                    print(text)
                    print('===' * 20)
                    print('\n')
    
    print('\n')  # Final newline
    return result