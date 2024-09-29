from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import stupid_test
import uuid

app = Flask(__name__)
api = Api(app)

# Пример списка "дублирующихся" видео (для демонстрации)
duplicates = {
    "https://example.com/video.mp4": "0003d59f-89cb-4c5c-9156-6c5bc07c6fad",
    "https://example.com/another_video.mp4": "000ab50a-e0bd-4577-9d21-f1f426144321",
}

class CheckVideoDuplicate(Resource):
    def post(self):
        data = request.get_json()
        if not data or 'link' not in data:
            return {'message': 'Неверный запрос'}, 400

        video_link = data['link']
        res = stupid_test.get_is_duplicate(video_link)
        print(res)
        # return {'is_duplicate': True, 'duplicate_for': ""}, 200
        if res[0]:
            return {
                'is_duplicate': True,
                'duplicate_for': res[1]
            }, 200

        return {
            'is_duplicate': False,
            'duplicate_for': None
        }, 200

api.add_resource(CheckVideoDuplicate, '/check-video-duplicate')

if __name__ == '__main__':
    app.run("0.0.0.0", port=5000)