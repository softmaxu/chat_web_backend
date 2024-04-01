import asyncio
import websockets
import json
import traceback
from llm_adapter import LLM_Adapter

model_dir="/data/usr/jy/Langchain-Chatchat/webui_pages/fine_tune/final_model/"
model_name="user_"+"模型2401"
system_msg="你是一个模具领域的AI助手MoldGPT，请根据上下文详细回答问题。"
rag_prompt="###请根据下列信息回答问题：@@@{rag_docs}###\n"
model=LLM_Adapter(model_dir, model_name, system_msg, rag_prompt)

async def send_ping(websocket):
    while True:
        await asyncio.sleep(10)  # 每10秒发送一次Ping
        if websocket.open:
            await websocket.ping()
            print("发送Ping")
        else:
            break

async def echo(websocket, path):
    # 启动发送心跳的任务
    ping_task = asyncio.create_task(send_ping(websocket))

    try:
        async for message in websocket:
            try:
                # 尝试解析JSON字符串
                message_json = json.loads(message)
                # print("message_json", message_json)
                position = 0
                await websocket.send(json.dumps({"text": "","operation":"syn"}))
                rag_to_user, llm_stream_output = model.predict(message_json)
                if rag_to_user:
                    await websocket.send(json.dumps({"text": rag_to_user}))
                    await websocket.send(json.dumps({"text": "","operation":"syn"}))
                for response in llm_stream_output:
                    sinppet = response[position:]
                    # print(sinppet, end='|', flush=True)
                    position = len(response)
                    response_message = json.dumps({"text": sinppet})
                    await websocket.send(response_message)
            except json.JSONDecodeError:
                # 如果解析失败，打印原始字符串
                print("解析失败：", message)
                response_message = "echo " + message
    except websockets.ConnectionClosedError as e:
        print("连接关闭：", e)
    except Exception as e:
        # 处理其他可能的异常
        print("发生错误：", e)
        traceback.print_exc()
    finally:
        # 取消心跳发送任务
        ping_task.cancel()
        try:
            # 等待任务被取消
            await ping_task
        except asyncio.CancelledError:
            # 任务取消时将抛出CancelledError异常
            print("Ping task cancelled")

async def main():
    address = "0.0.0.0"
    port = 8081
    print(f"Starting WebSocket server on ws://{address}:{port}")
    
    async with websockets.serve(echo, address, port):
        await asyncio.Future()

asyncio.run(main())
