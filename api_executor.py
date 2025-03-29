import httpx
import asyncio

class APIExecutor:
    def __init__(self, openapi_data):
        self.openapi_data = openapi_data
    
    async def execute_api(self, method, endpoint, payload=None):
        """Executes a given API request."""
        base_url = self.openapi_data.get("servers", [{}])[0].get("url", "")
        url = f"{base_url}{endpoint}"
        
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, json=payload)
            return response.json()
    
    async def run_load_test(self, num_users, duration):
        """Runs a load test with multiple users over a given duration."""
        async def single_user_test():
            return await self.execute_api("GET", list(self.openapi_data["paths"].keys())[0])
        
        tasks = [single_user_test() for _ in range(num_users)]
        results = await asyncio.gather(*tasks)
        return results
