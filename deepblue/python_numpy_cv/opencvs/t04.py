import cv2

cap = cv2.VideoCapture("./aa.mp4")
print("success")

for i in range(100):
    ret, frame = cap.read()
    cv2.imshow("arm", frame)
    cv2.imwrite(f"./info/{i}.jpg", frame)
cap.release()
cv2.destroyAllWindows()
