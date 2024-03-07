import asyncio
import websockets
import json
from llm_adapter import LLM_Adapter

model_dir="/data/usr/jy/Langchain-Chatchat/webui_pages/fine_tune/final_model/"
model_name="user_"+"模型2401"
model=LLM_Adapter(model_dir, model_name)

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
                parsed_json = json.loads(message)
                print(parsed_json)
                text=parsed_json["text"]
                print("收到：", text)
                response=model.predict(text)
                print(response)
                response_message = json.dumps({"text": "echo " + response})
            except json.JSONDecodeError:
                # 如果解析失败，打印原始字符串ß
                print("解析失败：", message)
                response_message = "echo " + message
            # 发送响应消息回客户端
            await websocket.send(response_message)
    except websockets.ConnectionClosedError as e:
        print("连接关闭：", e)
    except Exception as e:
        # 处理其他可能的异常
        print("发生错误：", e)
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
