import asyncio
import aiohttp

class APIExecutor:
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers

    async def execute_api(self, method, endpoint, payload=None):
        url = f"{self.base_url}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, json=payload, headers=self.headers) as response:
                return {"status_code": response.status, "response": await response.json()}

    async def run_load_test(self, api_sequence, users=100, duration=300):
        tasks = []
        for _ in range(users):
            for api in api_sequence:
                method, endpoint = api.split(" ", 1)
                payload = {"name": f"test-{_}"}  # Dynamic payloads
                tasks.append(self.execute_api(method, endpoint, payload))
        return await asyncio.gather(*tasks)
