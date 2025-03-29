class ResultStorage:
    def __init__(self):
        self.results = []

    def store(self, api, response):
        self.results.append({"api": api, "response": response})
