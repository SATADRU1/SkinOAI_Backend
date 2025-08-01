from roboflow import Roboflow

rf = Roboflow(api_key="K3NtQYY2EhfGyAd78fvK")
project = rf.workspace().project("my-first-project-apmvj")
model = project.version(1).model
result = model.predict("img1.jpg").json()
confidence=result['predictions'][0]['predictions'][0]['confidence']
class_=result['predictions'][0]['predictions'][0]['class']
print(class_,confidence)