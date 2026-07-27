import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f'indice {i}: no disponible')
        cap.release()
        continue
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(f'preview_{i}.jpg', frame)
        print(f'indice {i}: capturado, revisa preview_{i}.jpg')
    else:
        print(f'indice {i}: abierto pero sin frame')
    cap.release()
