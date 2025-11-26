import json

class LogRequestMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        print("----- Incoming Request -----")
        print("Method:", request.method)
        print("Path:", request.path)
        print("GET:", request.GET)
        print("POST:", request.POST)
        print("Headers:", dict(request.headers))
        if not request.content_type.startswith('multipart'):
            try:
                print("Body:", request.body.decode('utf-8'))
            except Exception as e:
                print("Could not print body:", e)
        else:
            print("Body: [multipart/form-data — skipped]")
        print("----------------------------")

        response = self.get_response(request)
        return response