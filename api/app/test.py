import requests

url = "http://localhost:5000/check-video-duplicate"
data = {
    "link": "https://s3.ritm.media/yappy-db-duplicates/2c29d956-9c02-483f-ae8a-d934b9dd1f41.mp4"
}
# print(test.get_is_duplicate(data['link']))
response = requests.post(url, json=data)

print(response.json())