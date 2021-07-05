# importacion de los paquetes necesarios
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2

def detect_people(frame, net, ln, personIdx=0):
	# tomar las dimensiones del marco e inicializar la lista de
	# resultados
	(H, W) = frame.shape[:2]
	results = []

	# construir un blob a partir del marco de entrada y luego realizar un avance
	# pase del detector de objetos YOLO, dándonos nuestros cuadros delimitadores
	# y probabilidades asociadas
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# inicializar nuestras listas de cuadros delimitadores, centroides y
	# confidencias, respectivamente
	boxes = []
	centroids = []
	confidences = []

	# bucle sobre cada una de las salidas de capa
	for output in layerOutputs:
		# recorrer cada una de las detecciones
		for detection in output:
			# extraer la identificación de la clase y la confianza (es decir, probabilidad)
			# de la detección de objetos actual
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filtrar las detecciones (1) asegurándose de que el objeto
			# detectado fue una persona y (2) que el mínimo
			# confianza se cumple
			if classID == personIdx and confidence > MIN_CONF:
				#escalar las coordenadas del cuadro delimitador en relación con
				# el tamaño de la imagen, teniendo en cuenta que YOLO
				# en realidad devuelve las coordenadas centrales (x, y) de
				# el cuadro delimitador seguido del ancho de los cuadros y
				# altura
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use las coordenadas del centro (x, y) para derivar la parte superior
				# y esquina izquierda del cuadro delimitador
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# actualice nuestra lista de coordenadas del cuadro delimitador,
				# centroides y confidencias
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

				# aplicar supresión no máxima para suprimir la superposición débil
				# cuadros delimitadores
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# asegúrese de que exista al menos una detección
	if len(idxs) > 0:
		# recorrer los índices que mantenemos
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# actualice nuestra lista de resultados para que consta de la persona
			# probabilidad de predicción, coordenadas del cuadro delimitador,
			# y el centroide
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# devuelve la lista de resultados
	return results