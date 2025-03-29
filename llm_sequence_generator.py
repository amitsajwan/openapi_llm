from openai import OpenAI

class LLMSequenceGenerator:
    def determine_intent(self, user_input, openapi_data):
        if "swagger" in user_input.lower() or "openapi" in user_input.lower():
            return "provide_openapi"
        elif "list" in user_input.lower() and "apis" in user_input.lower():
            return "list_apis"
        elif "run sequence" in user_input.lower():
            return "run_sequence"
        elif "load test" in user_input.lower():
            return "load_test"
        else:
            return "general_query"

    def suggest_sequence(self, openapi_data):
        return ["GET /users", "POST /users", "GET /users/{id}"]

    def extract_load_test_params(self, user_input):
        num_users = 10
        duration = 5
        if "users" in user_input and "minutes" in user_input:
            parts = user_input.split()
            for i, word in enumerate(parts):
                if word.isdigit():
                    num_users = int(word)
                if "minute" in parts[i + 1]:
                    duration = int(word)
        return num_users, duration
