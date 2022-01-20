# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import dlib
import pickle
from mtcnn.mtcnn import MTCNN
import cv2
from imutils import face_utils
import numpy as np
import pickle

# Load
filename = 'facemodels/model.sav'
clf = pickle.load(open(filename, 'rb'))

desc_file = "facemodels/face_desc.csv"
f = open(desc_file, "r")
desc = f.readlines()
f.close()
dict = {}
for line in desc:
    dict[line.split('|')[0]] = [line.split('|')[1],line.split('|')[2],line.split('|')[3]]

detector = MTCNN()
predictor = dlib.shape_predictor("facemodels/shape_predictor_68_face_landmarks.dat")


class ActionPix(Action):

    def name(self) -> Text:
        return "action_pix"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Xu ly hinh anh/nhan dien qua model
        image_url = tracker.latest_message["text"]
        if not image_url.startswith("http"):
            dispatcher.utter_message(text="Please send face photo to get recommendation!")
            return []


        # Luu hinh anh tu url ve
        import urllib.request
        import numpy as np
        import cv2
        resource = urllib.request.urlopen(image_url)
        image = np.asarray(bytearray(resource.read()), dtype="uint8")
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Nhan dien qua model de lay hinh dang khuon mat
        results = detector.detect_faces(frame)

        if len(results) != 0:
            for result in results:
                x1, y1, width, height = result['box']

                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                # Extract dlib
                landmark = predictor(frame, dlib.rectangle(x1, y1, x2, y2))
                landmark = face_utils.shape_to_np(landmark)

                print("O", landmark.shape)
                landmark = landmark.reshape(68 * 2)
                print("R", landmark.shape)

                # Co ket qua du doan
                y_pred = clf.predict([landmark])
                print(y_pred)

                face_desc = dict[y_pred[0]][1]
                face_shape = dict[y_pred[0]][0]
                face_image = dict[y_pred[0]][2]

                dispatcher.utter_message(
                    text="Bạn có KHUÔN {}.\nCách chọn kình phù hợp: {}".format(face_shape.upper(), face_desc),
                    image=face_image)
                dispatcher.utter_message(text="Please send face photo to get recommendation!")


        return []
