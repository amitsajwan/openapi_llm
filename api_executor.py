import requests
import asyncio
import random

class APIExecutor:
    async def execute_api(self, method, endpoint, payload=None):
        url = f"{self.base_url}{endpoint}"
        headers = self.headers
        response = requests.request(method, url, headers=headers, json=payload)
        return response.json()

    async def run_load_test(self, openapi_data, num_users, duration):
        endpoints = openapi_data.get_endpoints()
        results = []
        
        async def execute_random_api():
            endpoint = random.choice(endpoints)
            response = await self.execute_api("GET", endpoint)
            return response
        
        tasks = [execute_random_api() for _ in range(num_users)]
        results = await asyncio.gather(*tasks)
        return results
