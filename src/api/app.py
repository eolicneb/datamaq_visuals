import asyncio
import threading

from src.api.servers.buttler import update_buttler, server as buttler_server
from src.api.servers.band_position import process_band_edge, server as band_server


async def main():
    threading.Thread(target=update_buttler).start()
    threading.Thread(target=process_band_edge).start()

    await asyncio.gather(buttler_server.run_async(port=5000),
                         band_server.run_async(port=5001))


if __name__ == "__main__":
    asyncio.run(main())
